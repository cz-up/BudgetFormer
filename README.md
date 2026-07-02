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

Datasets are loaded from `./dataset/<name>/` (or wherever `--dataset_dir`
points); see [Data Preparation](#data-preparation) below for how each
dataset's raw files are obtained and converted.

---

## Data Preparation

All dataset preprocessing goes through `utils/preprocess_data.py`:

```bash
python utils/preprocess_data.py --dataset <name> [--split_id 0]
```

This writes `x.pt` / `edge_index.pt` / `y.pt` (and a split file, when
available) to `<dataset_dir>/<name>/` (`./dataset/<name>/` by default). It can
be run from any working directory — output always lands under
`<dataset_dir>/<name>/` relative to wherever you invoke it from, and raw
files are resolved with a fallback to `utils/dataset/<name>/` regardless of
cwd, so a fresh checkout only needs the raw files staged once.

### Auto-downloading datasets

`ogbn-arxiv`, `ogbn-products`, `ogbn-papers100M`, `cora`, `citeseer`,
`pubmed`, `dblp`, `CS`, `Physics`, `Photo`, `roman-empire`,
`amazon-ratings`, `minesweeper`, `tolokers`, `questions` fetch themselves via
OGB / PyTorch Geometric on first use — just run the command above.

### Datasets requiring raw files: snap-patents, genius, amazon

These ship as raw `.mat`/`.npz` files that must be staged locally before
running `preprocess_data.py`. Place them under `utils/dataset/` as shown
below (this is also where they already live in this checkout); the
processed output is regenerated from these on demand and is safe to delete.

| Dataset | Raw files | Source | Local path |
|---|---|---|---|
| snap-patents | `snap_patents.mat` | [LINKX / `CUAI/Non-Homophily-Large-Scale`, Google Drive](https://drive.google.com/file/d/1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia/view) (same file ID as `dataset_drive_url['snap-patents']` in that repo's `data_utils.py`) | `utils/dataset/snap_patents.mat` |
| genius | `genius.mat` | [LINKX / `CUAI/Non-Homophily-Large-Scale`](https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/data/genius.mat) — checked directly into that repo's `data/` dir (no Drive download step, unlike snap-patents) | `utils/dataset/genius.mat` |
| amazon | `adj_full.npz`, `feats.npy`, `labels.npy` | GraphSAINT / PyG `AmazonProducts` release (Google Drive file IDs `17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa`, `10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj`, and — see note below — `class_map.json` id `1LIl4kimLfftj4-7NmValuWyCQE8AaE7P`) | `utils/dataset/amazon/` |

> **amazon label format note:** `preprocess_data.py` expects a dense
> `labels.npy` (one-hot, `[num_nodes, num_classes]`), not the official
> `class_map.json` (a `{node_id: class_id}` dict). The `labels.npy` already
> in `utils/dataset/amazon/` was pre-converted from `class_map.json`; if you
> download the raw release fresh, convert `class_map.json` to that dense
> format yourself before running `preprocess_data.py`, or ask for that
> conversion to be added to the script.

Commands per dataset:

```bash
# snap-patents: raw tensors, then official label quantization (5-class,
# by publication year) + official train/valid/test split download
python utils/preprocess_data.py --dataset snap-patents
python utils/prepare_snap-patents.py --split_id 0   # needs `pip install gdown`

# genius: single command — generates x/edge_index/y plus 5 reproducible
# random 50/25/25 splits (LINKX convention, no official fixed split exists)
python utils/preprocess_data.py --dataset genius --split_id 0

# amazon: single command — generates x/edge_index/y and downloads the
# official GraphSAINT 80/5/15 split (role.json) automatically
python utils/preprocess_data.py --dataset amazon   # needs `pip install gdown`
```

`snap-patents` and `amazon` download their split files from Google Drive via
`gdown` on first run and cache them locally, so later runs (including in a
different `--dataset_dir`) don't re-download. `genius`'s splits are generated
locally with a fixed seed, so they're reproducible without network access.

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
