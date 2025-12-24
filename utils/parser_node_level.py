import argparse


def parser_add_main_args(parser):
   
    # main args
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--dataset', type=str, default='pubmed')

    # model args
    parser.add_argument('--model', type=str, default="graphormer")
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64) #128
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=1) # must match M power adj in preprocess_data
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_global_node', type=int, default=1)
    parser.add_argument('--attn_type', type=str, default="sparse", help='attention type: sparse/full/flash')
    parser.add_argument('--entropy_rank', type=int, default=0,
                        help='sequence-parallel rank used to log attention entropy (-1 disables logging)')

    # training args
    parser.add_argument('--seq_len', type=int, default=256000, help='total sequence length here')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_updates', type=int, default=10,
                        help='warmup steps for optimizer learning rate scheduling')
    parser.add_argument('--tot_updates',  type=int, default=70,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--epochs', type=int, default=1000) # larger seq len more training epochs
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    parser.add_argument('--peak_lr', type=float, default=2e-4)  
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perturb_feature', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--load_model', action='store_true', default=False, help='whether to load saved model')
    parser.add_argument('--model_dir', type=str, default='./model_ckpt/')
    parser.add_argument('--switch_freq', type=int, default=5)
    parser.add_argument('--subgraph_sampler', type=str, default='identity',
                        choices=['identity', 'edge_drop'],
                        help='子图采样器类型')
    parser.add_argument('--head_subgraph_provider', type=str, default='hop',
                        choices=['hop'],
                        help='为不同 head 提供不同子图的方式')
    parser.add_argument('--head_groups', type=int, default=4,
                        help='将所有 head 平分到的子图组数')
    parser.add_argument('--head_rw_walks_factor', type=float, default=1.0,
                        help='每节点随机游走次数=平均度*系数')
    parser.add_argument('--head_rw_length_factor', type=float, default=1.0,
                        help='随机游走长度=平均直径估计值*系数')
    parser.add_argument('--head_group_max_nodes', type=int, default=2048,
                        help='full_group 模式下每组最大节点数')
    
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

    parser.add_argument('--track_edge_coverage', action='store_true', default=False,
                        help='Report average retained-edge ratio for the first epoch.')
