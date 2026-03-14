def add_node_common_args(parser, defaults=None):
    defaults = defaults or {}

    parser.add_argument("--dataset_dir", type=str, default=defaults.get("dataset_dir", "./dataset/"))
    parser.add_argument("--dataset", type=str, default=defaults.get("dataset", "ogbn-arxiv"))

    parser.add_argument("--model", type=str, default=defaults.get("model", "graphormer"))
    parser.add_argument("--n_layers", type=int, default=defaults.get("n_layers", 4))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim", 64))
    parser.add_argument("--ffn_dim", type=int, default=defaults.get("ffn_dim", 256))
    parser.add_argument("--attn_bias_dim", type=int, default=defaults.get("attn_bias_dim", 1))
    parser.add_argument("--dropout_rate", type=float, default=defaults.get("dropout_rate", 0.3))
    parser.add_argument("--input_dropout_rate", type=float, default=defaults.get("input_dropout_rate", 0.1))
    parser.add_argument("--attention_dropout_rate", type=float, default=defaults.get("attention_dropout_rate", 0.5))
    parser.add_argument("--num_global_node", type=int, default=defaults.get("num_global_node", 1))
    parser.add_argument(
        "--attn_type",
        type=str,
        default=defaults.get("attn_type", "sparse"),
        help="attention type: sparse/full/flash",
    )

    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.01))
    parser.add_argument(
        "--warmup_updates",
        type=int,
        default=defaults.get("warmup_updates", 10),
        help="warmup steps for optimizer learning rate scheduling",
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 300))
    parser.add_argument("--eval_every", type=int, default=defaults.get("eval_every", 5))
    parser.add_argument("--peak_lr", type=float, default=defaults.get("peak_lr", 2e-4))
    parser.add_argument("--end_lr", type=float, default=defaults.get("end_lr", 1e-9))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--save_model", action="store_true", default=False, help="whether to save model")
    parser.add_argument("--model_dir", type=str, default=defaults.get("model_dir", "./model_ckpt/"))

    parser.add_argument(
        "--head_hop_walk_length",
        type=int,
        default=defaults.get("head_hop_walk_length", 4),
        help="head hop random-walk length",
    )
    parser.add_argument(
        "--head_hop_walks_per_node",
        type=int,
        default=defaults.get("head_hop_walks_per_node", 2),
        help="head hop random walks per node",
    )
    parser.add_argument(
        "--random_walk_device",
        type=str,
        default=defaults.get("random_walk_device", "same"),
        help="device for random-walk subgraph construction: same|cpu|cuda|cuda:N",
    )

    parser.add_argument("--rank", type=int, default=defaults.get("rank"))
    parser.add_argument("--local-rank", "--local_rank", type=int, default=defaults.get("local_rank"))
    parser.add_argument("--world-size", type=int, default=defaults.get("world_size"))
    parser.add_argument(
        "--distributed-backend",
        default=defaults.get("distributed_backend", "nccl"),
        choices=["nccl", "gloo", "ccl"],
        help="Which backend to use for distributed training.",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=defaults.get("distributed_timeout_minutes", 10),
        help="Timeout minutes for torch.distributed.",
    )
    parser.add_argument(
        "--sequence-parallel-size",
        type=int,
        default=defaults.get("sequence_parallel_size", 4),
        help="Enable DeepSpeed's sequence parallel.",
    )


def add_node_batch_sp_args(parser, defaults=None):
    defaults = defaults or {}

    parser.add_argument(
        "--batch_subgraph_mode",
        type=str,
        default=defaults.get("batch_subgraph_mode", "induced"),
        choices=["induced", "seed_rw"],
        help="batch subgraph construction: induced subgraph or seed-based full-graph random walk",
    )
    parser.add_argument(
        "--seed_batch_size",
        type=int,
        default=defaults.get("seed_batch_size"),
        help="number of seed/query nodes per batch when --batch_subgraph_mode seed_rw; defaults to seq_len",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=defaults.get("seq_len", 256000),
        help="total sequence length here",
    )

    parser.add_argument("--adaptive_walk", action="store_true", default=False, help="enable adaptive tuning of walk length/num walks")
    parser.add_argument("--adaptive_patience", type=int, default=defaults.get("adaptive_patience", 5), help="epochs without improvement before fixing L/R")
    parser.add_argument("--adaptive_eval_repeats", type=int, default=defaults.get("adaptive_eval_repeats", 3), help="repeat evaluation and average to reduce randomness")
    parser.add_argument("--adaptive_embed_batches", type=int, default=defaults.get("adaptive_embed_batches", 5), help="number of train batches to compare L vs L+1")
    parser.add_argument("--full_attn_hop_stats", action="store_true", default=False, help="enable full attention hop-mass stats in val")
    parser.add_argument("--full_attn_hop_mass", type=float, default=0.95, help="target hop mass ratio (e.g., 0.95)")
    parser.add_argument("--full_attn_hop_max_queries", type=int, default=defaults.get("full_attn_hop_max_queries", 64), help="max queries per batch for hop-mass stats")
    parser.add_argument("--full_attn_hop_max_batches", type=int, default=defaults.get("full_attn_hop_max_batches", 1), help="max batches per layer to collect hop-mass stats")
    parser.add_argument("--full_attn_hop_max_hop", type=int, default=defaults.get("full_attn_hop_max_hop", 64), help="max hop to explore when collecting hop-mass stats (<=0 for no limit)")
    parser.add_argument("--adaptive_cov_delta", type=float, default=defaults.get("adaptive_cov_delta", 0.03), help="min coverage improvement to reset patience")

    parser.add_argument(
        "--attn_bias_mode",
        type=str,
        default=defaults.get("attn_bias_mode", "none"),
        choices=["none", "local_spd"],
        help="Attention bias encoding mode: none (disabled) | local_spd (per-batch BFS)",
    )
    parser.add_argument("--attn_bias_max_dist", type=int, default=defaults.get("attn_bias_max_dist", 5), help="[local_spd] max hop distance; distances > this are clamped. one-hot dim = max_dist+1")


def add_node_fullgraph_sp_args(parser, defaults=None):
    defaults = defaults or {}

    parser.add_argument("--include_real_edges", type=int, default=defaults.get("include_real_edges", 0))
    parser.add_argument("--include_self_loops", type=int, default=defaults.get("include_self_loops", 0))
    parser.add_argument(
        "--to_bidirected",
        action="store_true",
        default=defaults.get("to_bidirected", False),
    )
    parser.add_argument("--seq_len", type=int, default=defaults.get("seq_len", 0), help="compat arg for shared launch scripts")


def normalize_main_node_batch_sp_args(args):
    return args


def normalize_main_node_fullgraph_sp_args(args):
    if args.model != "graphormer":
        args.num_global_node = 0
    elif args.num_global_node not in (0, 1):
        raise ValueError("main_node_fullgraph_sp.py currently supports at most one Graphormer virtual node.")
    return args


def parser_add_main_args(parser):
    add_node_common_args(parser)
    add_node_batch_sp_args(parser)
