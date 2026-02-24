import argparse


def parser_add_main_args(parser):
   
    # main args
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')

    # model args
    parser.add_argument('--model', type=str, default="graphormer")
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64) #128
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--attn_bias_dim', type=int, default=1) # must match M power adj in preprocess_data
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--attn_type', type=str, default="sparse", help='attention type: sparse/full/flash')

    # training args
    parser.add_argument('--seq_len', type=int, default=256000, help='total sequence length here')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_updates', type=int, default=10,
                        help='warmup steps for optimizer learning rate scheduling')
    parser.add_argument('--epochs', type=int, default=1000) # larger seq len more training epochs
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    parser.add_argument('--peak_lr', type=float, default=2e-4)  
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model_ckpt/')
    parser.add_argument('--head_hop_walk_length', type=int, default=4,
                        help='head hop 随机游走长度')
    parser.add_argument('--head_hop_walks_per_node', type=int, default=2,
                        help='head hop 随机游走次数/节点')
    parser.add_argument('--use_ogbn_split', type=int, default=0,
                        help='是否使用OGBN原始划分 (1启用, 0关闭)')
    parser.add_argument('--adaptive_walk', action='store_true', default=False,
                        help='enable adaptive tuning of walk length/num walks')
    parser.add_argument('--adaptive_patience', type=int, default=5,
                        help='epochs without improvement before fixing L/R')
    parser.add_argument('--adaptive_eval_repeats', type=int, default=3,
                        help='repeat evaluation and average to reduce randomness')
    parser.add_argument('--adaptive_embed_batches', type=int, default=5,
                        help='number of train batches to compare L vs L+1')
    parser.add_argument('--adaptive_embed_delta', type=float, default=0.05,
                        help='min relative output diff to keep increasing L')
    parser.add_argument('--full_attn_hop_stats', action='store_true', default=False,
                        help='enable full attention hop-mass stats in val')
    parser.add_argument('--full_attn_hop_mass', type=float, default=0.95,
                        help='target mass for hop-mass coverage stats')
    parser.add_argument('--full_attn_hop_max_queries', type=int, default=64,
                        help='max queries per batch for hop-mass stats')
    parser.add_argument('--full_attn_hop_max_batches', type=int, default=1,
                        help='max batches per layer to collect hop-mass stats')
    parser.add_argument('--full_attn_hop_max_hop', type=int, default=64,
                        help='max hop to explore when collecting hop-mass stats (<=0 for no limit)')
    parser.add_argument('--adaptive_val_delta', type=float, default=0.0,
                        help='min val acc improvement to reset patience')
    parser.add_argument('--adaptive_cov_delta', type=float, default=0.0,
                        help='min coverage improvement to reset patience')
    
    # distributed args
    parser.add_argument('--rank', type=int, default=None,
                       help='rank passed from distributed launcher.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    parser.add_argument('--world-size', type=int, default=None,
                       help='world size of sequence parallel group.')
    parser.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    parser.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    parser.add_argument('--sequence-parallel-size', type=int, default=4,
                       help='Enable DeepSpeed\'s sequence parallel.')
