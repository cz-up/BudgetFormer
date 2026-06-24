# <p align="center">BudgetFormer</p>

**BudgetFormer** is a distributed, full-graph Graph Transformer training system.
It scales Graph Transformers to large graphs by separating two questions and
solving them jointly:

1. **What attention to compute (logical).** Instead of dense all-pair attention,
   each query attends to a small, query-centric scope drawn from two sources
   under a fixed per-query budget *B* — **real graph edges** (deterministic local
   structure) and **random-walk edges** (budget-controlled multi-hop reach). A
   lightweight, probe-guided controller (**Adaptive Budget Selection**) allocates
   the budget between the two sources online, with no per-dataset tuning.

2. **How to execute it under memory limits (physical).** A profile-driven
   planner (**Multi-Tier Execution Planner**) compiles the chosen workload into a
   feasible, fast execution plan. Its primary lever is **memory-aware
   mixed-device edge construction**: when the random-walk construction would not
   fit GPU memory, it offloads *only* the random-walk build to the CPU while
   keeping real edges and topology on the GPU — staying within the HBM budget at
   nearly unchanged speed. Activation recomputation/retention is a secondary
   elastic reserve.

---

## Installation

We recommend a conda environment with Python 3.10:

```bash
conda create -n budgetformer python=3.10
conda activate budgetformer
pip install -r requirements.txt
```

Datasets (ogbn-arxiv, ogbn-products, Amazon, genius, snap-patents) are loaded
from `./dataset/`; OGB datasets download automatically on first use.

---

## Quick Start

Full-graph distributed training is launched with `torchrun` on
`main_node_fullgraph_sp.py`. Example: Graphormer on ogbn-arxiv, 4-way
sequence-parallel, adaptive budget + planner enabled:

```bash
torchrun --standalone --nproc_per_node=4 main_node_fullgraph_sp.py \
  --dataset ogbn-arxiv --model graphormer --n_layers 4 \
  --epochs 1000 --eval_every 10 --seed 42 \
  --sequence-parallel-size 4 \
  --walk_length 4 --walks_per_node 2 \
  --max_total_edges_per_query 8 \
  --adaptive_edge_budget \
  --activation_checkpoint_mode multi_tier
```

- `--model`: `graphormer`, `gt`, `graphgps`, or `exphormer`.
- `--sequence-parallel-size`: number of GPUs the sequence is split across
  (equal to `--nproc_per_node`).

---

## Key Hyperparameters

### Adaptive Budget Selection (the logical scope)

| Flag | Meaning | Default |
|---|---|---|
| `--adaptive_edge_budget` | Enable online real/RW budget selection | off |
| `--max_total_edges_per_query` | Total per-query edge budget *B* | 0 |
| `--walk_length` | Random-walk length *L* | 4 |
| `--walks_per_node` | Walks per node *P* (controls RW coverage) | 2 |

The controller probes neighboring `(real, rw)` allocations on the validation
subset and freezes a stable split `(b_r^*, b_w^*)` once it stops improving.

To use a **fixed** split instead of adaptive selection, drop
`--adaptive_edge_budget` and set:

| Flag | Meaning |
|---|---|
| `--fixed_real_edges_per_query` | Real-edge budget *b_r* |
| `--fixed_rw_edges_per_query` | Random-walk budget *b_w* |
| `--fixed_walk_length` | Walk length for the fixed split |

### Multi-Tier Execution Planner (the physical plan)

| Flag | Meaning | Default |
|---|---|---|
| `--activation_checkpoint_mode` | `multi_tier` enables the planner; also `all` / `ffn_only` for fixed recomputation | none |
| `--multi_tier_gpu_memory_limit_mib` | HBM budget the planner must fit within (MiB) | 0 (physical) |
| `--edge_build_device` | Default edge-build device `gpu`/`cpu` (the planner may offload RW to CPU automatically) | gpu |
| `--disable_edge_prefetch` | Ablation: build next-epoch edges in the foreground (no CPU↔GPU overlap) | off |
| `--force_multi_tier_plan` | Ablation: force a specific `edge_policy:tier_config` instead of letting the planner choose | "" |

When `--activation_checkpoint_mode multi_tier` is set, the planner profiles the
realized workload and selects, under the memory limit, both the
edge-construction policy (GPU-persistent / GPU-ephemeral / CPU rank-local
prefetch / CPU broadcast prefetch) and the activation-retention level. The
chosen plan is printed as, e.g.:

```
[MultiTierManager] Synced ACTIVE plan: edge_policy=gpu_persist tiers={'recompute': 2, 'retain': 2} placement=(real=GPU, rw=GPU(->CPU if oversized), topology=GPU)
```

---

## Reproducing Experiments

Helper scripts live under `scripts/`:

- `scripts/planner_vs_fixed_policy.sh` — planner vs. fixed execution strategies
  across memory budgets (Section "Effectiveness of the planner").
- `scripts/graphormer_hparam_influence.sh` — the `L × P` hyperparameter sweep.
- `scripts/edge_budget_sweep.sh` — budget-related sweeps.

Example — the planner / memory-budget sweep on Amazon and snap-patents:

```bash
MODEL=graphormer DATASETS="amazon snap-patents" \
  CONDITIONS="naive_aggressive naive_gpu_recompute naive_conservative planner" \
  MEM_LIMITS_MIB="16000 24000 40000" \
  BUDGET=8 SPLITS="4,4" LARGE_BUDGET=16 \
  bash scripts/ch4_naive_fixed_policy.sh
```

Plotting utilities (heatmaps, split-fraction / split-bar figures) are under
`plot/`.
