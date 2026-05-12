"""GraphGPS Transformer-only adapted for the distributed full-graph SP training framework.

GPS Transformer sub-module without the MPNN component (Rampasek et al., 2022).

Supported attn_type values
  sparse   : sparse scatter-aggregate over real/RW edges (default; scalable to large graphs)
  full     : dense O(N²) softmax attention — matches GPS 'Transformer' global model exactly
  performer: FAVOR+ linear O(N) approximation — matches GPS 'Performer' global model exactly

Architecture vs. Exphormer:
  - No edge-type modulation (no expander edges, no virtual node).
  - Q/K/V bias and out_proj follow each attention mode's original GPS defaults.
  - Post-norm BatchNorm, ReLU FFN with 2×dim_h hidden (matching GPS gps_layer.py).

References:
  GPS : Recipe for a General, Powerful, Scalable Graph Transformer (Rampasek et al., 2022)
        https://arxiv.org/abs/2205.12454
  FAVOR+: Rethinking Attention with Performers (Choromanski et al., 2021)
        https://arxiv.org/abs/2009.14794
  FAVOR+ code adapted from performer-pytorch by lucidrains (MIT License)
        https://github.com/lucidrains/performer-pytorch
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter

from gt_sp.initialize import (
    get_sequence_parallel_group,
    sequence_parallel_is_initialized,
)
from gt_sp.gt_layer import DistributedAttentionNodeLevel
from gt_sp.multi_tier import _MultiTierResourceManager, apply_tier


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# FAVOR+ helpers — adapted from performer-pytorch (lucidrains, MIT License)
# ---------------------------------------------------------------------------

def _gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None):
    """Random orthogonal projection matrix for FAVOR+."""
    nb_full_blocks = nb_rows // nb_columns
    block_list = []
    for _ in range(nb_full_blocks):
        q, _ = torch.linalg.qr(torch.randn(nb_columns, nb_columns), mode="reduced")
        block_list.append(q.t())
    remaining = nb_rows - nb_full_blocks * nb_columns
    if remaining > 0:
        q, _ = torch.linalg.qr(torch.randn(nb_columns, nb_columns), mode="reduced")
        block_list.append(q.t()[:remaining])
    mat = torch.cat(block_list, dim=0)                           # (nb_rows, nb_columns)
    multiplier = torch.randn(nb_rows, nb_columns).norm(dim=1)    # isotropic scaling
    return torch.diag(multiplier) @ mat


def _softmax_kernel(data, *, projection_matrix, is_query, eps=1e-4):
    """FAVOR+ softmax kernel feature map.  data: (B, H, N, d)."""
    b, h, *_ = data.shape
    d = data.shape[-1]
    data_normalizer = d ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5

    # project: (B, H, N, nb_features)
    proj = projection_matrix.type_as(data)                        # (r, d)
    proj = proj.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)   # (B, H, r, d)
    data_dash = torch.einsum("...nd,...rd->...nr", data_normalizer * data, proj)

    diag_data = (data ** 2).sum(-1, keepdim=True) * 0.5 * (data_normalizer ** 2)

    if is_query:
        data_dash = ratio * torch.exp(
            data_dash - diag_data - data_dash.amax(dim=-1, keepdim=True)
        ).clamp(min=eps)
    else:
        data_dash = ratio * torch.exp(
            data_dash - diag_data - data_dash.amax(dim=(-1, -2), keepdim=True)
        ).clamp(min=eps)
    return data_dash.type_as(data)


def _linear_attention(q, k, v):
    """Non-causal linear attention.  q/k: (B, H, N, r), v: (B, H, N, d)."""
    k_sum = k.sum(dim=-2)                                          # (B, H, r)
    D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_sum)      # (B, H, N)
    context = torch.einsum("...nd,...ne->...de", k, v)             # (B, H, r, d)
    out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)  # (B, H, N, d)
    return out


class PerformerFastAttention(nn.Module):
    """FAVOR+ kernel attention module (stateless except for the projection buffer).

    Matches the FastAttention used inside performer_pytorch.SelfAttention.
    The projection_matrix is registered as a buffer so it is broadcast/synced
    across SP ranks via sync_params_and_buffers at startup.
    """

    def __init__(self, dim_heads, nb_features=None):
        super().__init__()
        nb_features = nb_features or max(1, int(dim_heads * math.log(dim_heads)))
        proj = _gaussian_orthogonal_random_matrix(nb_features, dim_heads)
        self.register_buffer("projection_matrix", proj)

    def redraw_projection_matrix(self, device=None):
        """Re-randomize the projection matrix (call periodically for variance reduction)."""
        nb_rows, nb_columns = self.projection_matrix.shape
        proj = _gaussian_orthogonal_random_matrix(nb_rows, nb_columns,
                                                  device=device or self.projection_matrix.device)
        self.projection_matrix.copy_(proj)

    def forward(self, q, k, v):
        """q / k / v: (B, H, N, d_head)  →  output (B, H, N, d_head)."""
        q_prime = _softmax_kernel(q, projection_matrix=self.projection_matrix, is_query=True)
        k_prime = _softmax_kernel(k, projection_matrix=self.projection_matrix, is_query=False)
        return _linear_attention(q_prime, k_prime, v)


# ---------------------------------------------------------------------------
# Core attention — dispatches sparse / full / performer
# ---------------------------------------------------------------------------

class GPSCoreAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            from gt_sp.initialize import get_sequence_parallel_world_size
            seq_parallel_world_size = get_sequence_parallel_world_size()

        self.hidden_size_per_attention_head = hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // seq_parallel_world_size
        self.num_heads = num_heads
        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.performer = PerformerFastAttention(self.hidden_size_per_attention_head)

    def _sparse(self, q, k, v, edge_index, num_heads, head_dim):
        """Scatter-aggregate sparse attention over edge_index."""
        batch_size, s_len = q.size(0), q.size(1)
        if isinstance(edge_index, torch.Tensor) and edge_index.size(0) == 3:
            edge_index = edge_index[:2]

        q_flat = q.view(-1, num_heads, head_dim)
        k_flat = k.view(-1, num_heads, head_dim)
        v_flat = v.view(-1, num_heads, head_dim)

        src  = k_flat[edge_index[0].long()]
        dest = q_flat[edge_index[1].long()]
        score = (torch.mul(src, dest) / self.scale).sum(-1, keepdim=True).clamp(-5, 5)
        score = torch.exp(score)

        msg = v_flat[edge_index[0].long()] * score
        wV = torch.zeros_like(v_flat)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce="add")
        Z = score.new_zeros(v_flat.size(0), num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=Z, reduce="add")
        return (wV / (Z + 1e-6)).view(batch_size, s_len, -1)

    def _full(self, q, k, v):
        """Dense O(N²) softmax attention.  q/k/v: (B, N, H, d)."""
        batch_size, s_len = q.size(0), q.size(1)
        q = q.transpose(1, 2)                  # (B, H, N, d)
        k = k.transpose(1, 2).transpose(2, 3)  # (B, H, d, N)
        v = v.transpose(1, 2)
        attn = torch.softmax(torch.matmul(q / self.scale, k), dim=-1)
        attn = self.att_dropout(attn)
        x = attn.matmul(v).transpose(1, 2).contiguous()  # (B, N, H, d)
        return x.view(batch_size, s_len, -1)

    def _performer(self, q, k, v):
        """FAVOR+ linear attention.  q/k/v: (B, N, H, d)."""
        batch_size, s_len = q.size(0), q.size(1)
        q = q.transpose(1, 2)  # (B, H, N, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = self.performer(q, k, v)            # (B, H, N, d)
        return x.transpose(1, 2).contiguous().view(batch_size, s_len, -1)

    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        num_heads = self.num_attention_heads_per_partition if self.training else self.num_heads
        head_dim = self.hidden_size_per_attention_head
        attn_type = "sparse" if attn_type is None else str(attn_type).lower()

        if attn_type == "full":
            return self._full(q, k, v)
        if attn_type == "performer":
            return self._performer(q, k, v)
        return self._sparse(q, k, v, edge_index, num_heads, head_dim)


# ---------------------------------------------------------------------------
# Multi-head attention wrapper
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_type="sparse"):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        # Q/K/V bias follows each mode's GPS original:
        #   full     → bias=True  (torch.nn.MultiheadAttention default)
        #   performer → bias=False (performer_pytorch SelfAttention qkv_bias=False default)
        #   sparse   → bias=False (Exphormer style)
        qkv_bias = (str(attn_type).lower() == "full")
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size, bias=qkv_bias)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size, bias=qkv_bias)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)  # out_proj bias=True in all modes

        local_attn = GPSCoreAttention(hidden_size, attention_dropout_rate, num_heads)
        if sequence_parallel_is_initialized():
            self.dist_attn = DistributedAttentionNodeLevel(
                local_attn, get_sequence_parallel_group()
            )
        else:
            self.dist_attn = local_attn

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        orig_size = x.size()
        batch_size = x.size(0)
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)
        x = self.out_proj(x)
        assert x.size() == orig_size
        return x


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, num_heads, attn_type="sparse"):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_type=attn_type
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.BatchNorm1d(hidden_size)

        self.ffn1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ffn2 = nn.Linear(hidden_size * 2, hidden_size)
        self.ff_dropout1 = nn.Dropout(dropout_rate)
        self.ff_dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.BatchNorm1d(hidden_size)

    def _bn(self, norm, x):
        b, n, c = x.shape
        return norm(x.view(b * n, c)).view(b, n, c)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        h = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        h = self.dropout1(h)
        x = self._bn(self.norm1, x + h)
        h = self.ff_dropout1(F.relu(self.ffn1(x)))
        h = self.ff_dropout2(self.ffn2(h))
        x = self._bn(self.norm2, x + h)
        return x

    def forward_attn_only(self, x, attn_bias=None, edge_index=None, attn_type=None, **kwargs):
        h = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        h = self.dropout1(h)
        return self._bn(self.norm1, x + h)

    def forward_ffn_checkpointed(self, x):
        def _ffn(h):
            out = self.ff_dropout1(F.relu(self.ffn1(h)))
            out = self.ff_dropout2(self.ffn2(out))
            return self._bn(self.norm2, h + out)
        return checkpoint(_ffn, x, use_reentrant=False)


# ---------------------------------------------------------------------------
# Top-level GraphGPS model
# ---------------------------------------------------------------------------

class GraphGPS(nn.Module):
    """GraphGPS Transformer sub-module (no MPNN) for node-level distributed SP training.

    attn_type is fixed at construction and controls both the attention kernel and
    the Q/K/V projection bias to match the original GPS implementation.
    """

    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        input_dropout_rate,
        attention_dropout_rate,
        ffn_dim,
        num_global_node,   # interface compatibility, not used
        attn_type="sparse",
        **kwargs,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, dropout_rate, attention_dropout_rate, num_heads,
                         attn_type=attn_type)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.apply(lambda m: init_params(m, n_layers=n_layers))

        self.activation_checkpoint = False
        self.activation_checkpoint_mode = "none"
        self._comm_ckpt = None

    def set_activation_checkpoint(self, enabled: bool = True, mode: str | None = None, deferred: bool = False) -> None:
        mode_aliases = {
            None: "none",
            "none": "none",
            "all": "layer",
            "layer": "layer",
            "ffn_only": "ffn_only",
            "multi_tier": "multi_tier",
        }
        resolved_mode = mode_aliases.get(mode)
        if resolved_mode is None:
            raise ValueError(
                f"activation_checkpoint_mode must be 'all', 'ffn_only', or 'multi_tier', got {mode!r}"
            )
        enabled = bool(enabled) and resolved_mode != "none"
        self.activation_checkpoint = enabled
        self.activation_checkpoint_mode = resolved_mode if enabled else "none"
        if self.activation_checkpoint_mode == "multi_tier":
            self._comm_ckpt = _MultiTierResourceManager(len(self.layers), deferred=deferred)
        else:
            self._comm_ckpt = None

    def comm_aware_notify_budget_frozen(self, reuse_deferred_baseline: bool = False) -> None:
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_budget_frozen(reuse_deferred_baseline=reuse_deferred_baseline)

    def comm_aware_notify_step_end(self, device, t_bwd: float = None, t_fwd: float = None) -> None:
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_step_end(device, t_bwd=t_bwd, t_fwd=t_fwd)

    def redraw_performer_projections(self, device=None):
        """Re-randomize all FAVOR+ projection matrices (call periodically during training)."""
        for layer in self.layers:
            core = layer.self_attention.dist_attn
            core = getattr(core, "local_attn", core)
            if hasattr(core, "performer"):
                core.performer.redraw_projection_matrix(device=device)

    def forward(self, x, attn_bias, edge_index, attn_type=None):
        x = x.unsqueeze(0)
        x = self.input_dropout(self.node_encoder(x))

        use_ckpt = self.activation_checkpoint and self.training and torch.is_grad_enabled()
        ckpt_mode = getattr(self, "activation_checkpoint_mode", "none")

        if use_ckpt and ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
            self._comm_ckpt.plan(x.device)

        for i, layer in enumerate(self.layers):
            if not use_ckpt:
                x = layer(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
                continue
            if ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
                layer_mode = self._comm_ckpt.mode(i)
            else:
                layer_mode = ckpt_mode
            x = apply_tier(layer, layer_mode, x, attn_bias=attn_bias, edge_index=edge_index, attn_type=attn_type)

        if use_ckpt and ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
            self._comm_ckpt.record_post_forward_memory(x.device)

        return F.log_softmax(self.output_proj(x[0]), dim=1)
