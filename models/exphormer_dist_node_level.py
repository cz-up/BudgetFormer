"""Exphormer adapted for the distributed full-graph SP training framework.

Aligned with the original Exphormer (Shirzad et al., 2023) Transformer architecture:
  - Post-norm BatchNorm (no LayerNorm).
  - ReLU activation in FFN with hidden size 2×dim_h (matching MultiLayer in ../Exphormer/).
  - No Q/K/V output projection bias (bias=False), no O projection.
  - Edge encoding: Embedding(N_EDGE_TYPES, dim_h) → Linear(dim_h, dim_h, bias=False),
    matching original DummyEdge → nn.Linear pipeline.  Type 0 = real/RW edges
    (equivalent to DummyEdge constant), type 1 = expander edges (innovation).
  - Expander edges generated once at startup and merged per epoch by main script.

Reference: Exphormer (Shirzad et al., 2023), https://arxiv.org/abs/2303.06147
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
# Core attention
# ---------------------------------------------------------------------------

class ExphormerCoreAttention(nn.Module):
    """Sparse attention with optional edge-type modulation.

    If edge_index has a 3rd row (size(0)==3), row-2 values are treated as
    integer edge types fed through edge_encoder to produce per-head, per-dim
    modulation vectors E.  The attention score then becomes:
        score = (Q * K * E).sum(-1) / sqrt(d)
    instead of the plain (Q * K).sum(-1) / sqrt(d).
    """

    N_EDGE_TYPES = 2  # 0 = real/RW, 1 = expander

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

        # Edge encoding matching original: Embedding(type) → Linear → (num_heads, head_dim).
        # Type 0 = real/RW (equivalent to original DummyEdge constant per edge),
        # type 1 = expander (innovation). Linear has bias=False matching original E proj.
        hidden_size = num_heads * self.hidden_size_per_attention_head
        self.edge_type_emb = nn.Embedding(self.N_EDGE_TYPES, hidden_size)
        self.edge_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        batch_size, s_len = q.size(0), q.size(1)
        num_heads = q.size(2)
        head_dim = self.hidden_size_per_attention_head

        # Extract edge type from row-2 if present
        edge_type = None
        if isinstance(edge_index, torch.Tensor) and edge_index.size(0) == 3:
            edge_type = edge_index[2].long()
            edge_index = edge_index[:2]

        q_flat = q.view(-1, num_heads, head_dim)
        k_flat = k.view(-1, num_heads, head_dim)
        v_flat = v.view(-1, num_heads, head_dim)

        src = k_flat[edge_index[0].long()]   # (E, heads, hn)
        dest = q_flat[edge_index[1].long()]  # (E, heads, hn)
        score = torch.mul(src, dest) / self.scale  # (E, heads, hn)

        # Exphormer edge-feature modulation (Embedding → Linear, matching original E pipeline)
        if edge_type is not None:
            E = self.edge_proj(self.edge_type_emb(edge_type)).view(-1, num_heads, head_dim)
            score = torch.mul(score, E)

        score = score.sum(-1, keepdim=True).clamp(-5, 5)
        score = torch.exp(score)  # (E, heads, 1)

        msg = v_flat[edge_index[0].long()] * score
        wV = torch.zeros_like(v_flat)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce="add")

        Z = score.new_zeros(v_flat.size(0), num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=Z, reduce="add")

        x = wV / (Z + 1e-6)
        x = x.view(batch_size, s_len, -1)
        return x


# ---------------------------------------------------------------------------
# Multi-head attention wrapper (mirrors GT's, uses distributed attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size, bias=False)

        local_attn = ExphormerCoreAttention(hidden_size, attention_dropout_rate, num_heads)
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
        assert x.size() == orig_size
        return x


# ---------------------------------------------------------------------------
# Encoder layer: Post-norm + BatchNorm + ReLU (matching original Exphormer MultiLayer)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.BatchNorm1d(hidden_size)

        self.ffn1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ffn2 = nn.Linear(hidden_size * 2, hidden_size)
        self.ff_dropout1 = nn.Dropout(dropout_rate)  # after activation, before ffn2
        self.ff_dropout2 = nn.Dropout(dropout_rate)  # after ffn2, before residual
        self.norm2 = nn.BatchNorm1d(hidden_size)

    def _bn(self, norm, x):
        """Apply BatchNorm1d to (B, N, C) by temporarily flattening the batch dim."""
        b, n, c = x.shape
        return norm(x.view(b * n, c)).view(b, n, c)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        # Post-norm attention block (matching original Exphormer: residual → BatchNorm, no O proj)
        h = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        h = self.dropout1(h)
        x = self._bn(self.norm1, x + h)

        # Post-norm FFN block (ReLU, two dropouts matching original MultiLayer._ff_block)
        h = self.ff_dropout1(F.relu(self.ffn1(x)))
        h = self.ff_dropout2(self.ffn2(h))
        x = self._bn(self.norm2, x + h)
        return x

    def forward_attn_only(self, x, attn_bias=None, edge_index=None, attn_type=None, **kwargs):
        """MHA sub-block only (used by apply_tier keep_mha path)."""
        h = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        h = self.dropout1(h)
        return self._bn(self.norm1, x + h)

    def forward_ffn_checkpointed(self, x):
        """FFN sub-block with gradient checkpointing (used by apply_tier keep_mha path)."""
        def _ffn(h):
            out = self.ff_dropout1(F.relu(self.ffn1(h)))
            out = self.ff_dropout2(self.ffn2(out))
            return self._bn(self.norm2, h + out)
        return checkpoint(_ffn, x, use_reentrant=False)


# ---------------------------------------------------------------------------
# Top-level Exphormer model
# ---------------------------------------------------------------------------

class Exphormer(nn.Module):
    """Exphormer for node-level tasks in the distributed full-graph SP framework.

    forward() interface is identical to GT:
        forward(x_local, attn_bias, edge_index, attn_type=None)

    When edge_index has a 3rd row, it is treated as edge type (0=real/RW, 1=expander)
    and used for edge-feature-modulated attention.
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
        num_global_node,   # accepted for interface compatibility, not used
        **kwargs,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, dropout_rate, attention_dropout_rate, num_heads)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.apply(lambda m: init_params(m, n_layers=n_layers))

        self.activation_checkpoint = False
        self.activation_checkpoint_mode = "none"
        self._comm_ckpt = None  # _MultiTierResourceManager | None

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

    def forward(self, x, attn_bias, edge_index, attn_type=None):
        x = x.unsqueeze(0)                        # (1, N_local, d)
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
