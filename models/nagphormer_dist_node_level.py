"""NAGphormer adapted for the distributed full-graph SP training framework.

Key differences from GT/Graphormer:
  - Input x_local: (N_local, hops+1, input_dim) — pre-computed multi-hop features.
  - No cross-node attention: the Transformer attends within each node's K-hop sequence
    independently, so no sequence-parallel AllToAll is needed.
  - edge_index and attn_bias are accepted but ignored; NAGphormer uses only the
    pre-aggregated hop features computed by _compute_multihop_features().
  - Gradient synchronisation via the existing grad_reducer still applies.

Reference: NAGphormer (Chen et al., 2023), https://arxiv.org/abs/2206.04910
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(self, q, k, v):
        # q, k, v: (N_local, seq_len, hidden)
        orig_size = q.size()
        bsz, seq_len, _ = q.size()
        d = self.att_size

        q = self.linear_q(q).view(bsz, seq_len, self.num_heads, d).transpose(1, 2)
        k = self.linear_k(k).view(bsz, seq_len, self.num_heads, d).transpose(1, 2)
        v = self.linear_v(v).view(bsz, seq_len, self.num_heads, d).transpose(1, 2)

        # (bsz, heads, seq, seq)
        attn = torch.matmul(q * self.scale, k.transpose(2, 3))
        attn = torch.softmax(attn, dim=-1)
        attn = self.att_dropout(attn)
        x = attn.matmul(v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        x = self.output_layer(x)
        assert x.size() == orig_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pre-norm residual transformer (matching original NAGphormer)
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class NAGphormer(nn.Module):
    """NAGphormer for node-level tasks in the distributed full-graph SP framework.

    Expects x_local of shape (N_local, hops+1, input_dim) — multi-hop features
    pre-computed by _compute_multihop_features() before training.
    attn_bias and edge_index arguments are accepted for interface compatibility
    but are not used.
    """

    def __init__(
        self,
        hops,
        n_class,
        input_dim,
        n_layers=1,
        num_heads=8,
        hidden_dim=512,
        ffn_dim=None,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        **kwargs,  # absorb extra kwargs for interface compatibility
    ):
        super().__init__()
        self.seq_len = hops + 1
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        if ffn_dim is None or ffn_dim <= 0:
            ffn_dim = 2 * hidden_dim

        self.att_embeddings_nope = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Hop-attention aggregation head (matches original NAGphormer)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)
        self.Linear1 = nn.Linear(hidden_dim // 2, n_class)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        self.apply(lambda m: init_params(m, n_layers=n_layers))

    def forward(self, x_local, attn_bias, edge_index, attn_type=None):
        # x_local: (N_local, hops+1, input_dim) — pre-computed multi-hop features
        # attn_bias, edge_index: not used by NAGphormer
        tensor = self.att_embeddings_nope(x_local)  # (N_local, seq_len, hidden_dim)

        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        output = self.final_ln(tensor)

        # Hop-attention: weight neighbors by learned scores against central node
        node_tensor = output[:, :1, :]          # (N_local, 1,    hidden_dim)
        neighbor_tensor = output[:, 1:, :]      # (N_local, hops, hidden_dim)
        target = output[:, 0:1, :].expand(-1, self.seq_len - 1, -1)

        layer_atten = self.attn_layer(torch.cat([target, neighbor_tensor], dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_agg = (neighbor_tensor * layer_atten).sum(dim=1, keepdim=True)
        out = (node_tensor + neighbor_agg).squeeze(1)  # (N_local, hidden_dim)

        out = self.Linear1(F.relu(self.out_proj(out)))
        return F.log_softmax(out, dim=1)
