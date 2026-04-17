import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


def _mask_to_index(mask: torch.Tensor, split_id: int = 0) -> torch.Tensor:
    """Convert 1D/2D split masks to node indices."""
    if mask.dim() == 2:
        if split_id < 0 or split_id >= mask.size(1):
            raise ValueError(f"split_id={split_id} out of range for mask with {mask.size(1)} splits")
        mask = mask[:, split_id]
    return torch.nonzero(mask.to(torch.bool), as_tuple=True)[0]


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, n_layers, dropout=0.5, aggregator_type='mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        # input layer
        self.layers.append(dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(dglnn.SAGEConv(hidden_feats, hidden_feats, aggregator_type))
        # output layer
        self.layers.append(dglnn.SAGEConv(hidden_feats, out_feats, aggregator_type))

    def forward(self, g, features):
        is_blocks = isinstance(g, list) or isinstance(g, tuple)
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            block = g[i] if is_blocks else g
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = F.relu(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, n_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        # input layer
        self.layers.append(dglnn.GraphConv(in_feats, hidden_feats))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(dglnn.GraphConv(hidden_feats, hidden_feats))
        # output layer
        self.layers.append(dglnn.GraphConv(hidden_feats, out_feats))

    def forward(self, g, features):
        is_blocks = isinstance(g, list) or isinstance(g, tuple)
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            block = g[i] if is_blocks else g
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = F.relu(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, n_layers, dropout=0.5, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        # input layer
        self.layers.append(dglnn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.elu))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(dglnn.GATConv(hidden_feats * num_heads, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout, activation=F.elu))
        # output layer
        self.layers.append(dglnn.GATConv(hidden_feats * num_heads, out_feats, 1, feat_drop=dropout, attn_drop=dropout, activation=None))

    def forward(self, g, features):
        is_blocks = isinstance(g, list) or isinstance(g, tuple)
        h = features
        for i, layer in enumerate(self.layers):
            block = g[i] if is_blocks else g
            h = layer(block, h)
            if i != self.n_layers - 1:
                # GAT output has shape (N, num_heads, out_dim), flatten it
                h = h.flatten(1)
            else:
                # Output layer usually produces (N, 1, out_classes) -> squeeze to (N, out_classes)
                h = h.mean(1) if h.dim() == 3 else h
        return h

def load_dataset(dataset_name, root="./dataset", split_id: int = 0):
    print(f"Loading dataset {dataset_name}...")
    
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        from dgl.data import CitationGraphDataset
        data = CitationGraphDataset(name=dataset_name, raw_dir=root)
        g = data[0]
        num_classes = data.num_classes
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        labels = g.ndata['label']
        features = g.ndata['feat']
        train_idx = _mask_to_index(train_mask, split_id=split_id)
        val_idx = _mask_to_index(val_mask, split_id=split_id)
        test_idx = _mask_to_index(test_mask, split_id=split_id)

    elif dataset_name == 'reddit':
        from dgl.data import RedditDataset
        data = RedditDataset(raw_dir=root)
        g = data[0]
        num_classes = data.num_classes
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        labels = g.ndata['label']
        features = g.ndata['feat']
        train_idx = _mask_to_index(train_mask, split_id=split_id)
        val_idx = _mask_to_index(val_mask, split_id=split_id)
        test_idx = _mask_to_index(test_mask, split_id=split_id)

    elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        from ogb.nodeproppred import DglNodePropPredDataset
        data = DglNodePropPredDataset(name=dataset_name, root=root)
        g, labels = data[0]
        num_classes = data.num_classes
        labels = labels.squeeze()
        features = g.ndata['feat']
        
        # OGB split
        split_idx = data.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

    elif dataset_name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        from torch_geometric.datasets import HeterophilousGraphDataset
        pyg_name = dataset_name.capitalize()
        pyg_data = HeterophilousGraphDataset(root=root, name=pyg_name)[0]

        src = pyg_data.edge_index[0].to(torch.long)
        dst = pyg_data.edge_index[1].to(torch.long)
        num_nodes = int(pyg_data.x.size(0))
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        features = pyg_data.x.to(torch.float32)
        labels = pyg_data.y.to(torch.long).view(-1)
        num_classes = int(labels.max().item()) + 1

        if not hasattr(pyg_data, "train_mask") or not hasattr(pyg_data, "val_mask") or not hasattr(pyg_data, "test_mask"):
            raise ValueError(f"{pyg_name} dataset does not provide default train/val/test masks.")
        train_idx = _mask_to_index(pyg_data.train_mask, split_id=split_id)
        val_idx = _mask_to_index(pyg_data.val_mask, split_id=split_id)
        test_idx = _mask_to_index(pyg_data.test_mask, split_id=split_id)
        
    elif dataset_name == 'snap-patents':
        import os
        data_path = os.path.join(root, dataset_name)
        features = torch.load(f"{data_path}/x.pt", map_location="cpu", weights_only=False)
        labels = torch.load(f"{data_path}/y.pt", map_location="cpu", weights_only=False).long().view(-1)
        edge_index = torch.load(f"{data_path}/edge_index.pt", map_location="cpu", weights_only=False)
        num_nodes = features.shape[0]

        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        num_classes = int(labels.max().item()) + 1

        from utils.split_utils import load_default_split
        split_idx = load_default_split(dataset_name, root, split_id=split_id)
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    # ensure it's a homogeneous graph and add self loops securely
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)

    return g, features, labels, num_classes, train_idx, val_idx, test_idx


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main():
    parser = argparse.ArgumentParser(description='GNN Node Classification using DGL')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['cora', 'citeseer', 'pubmed', 'reddit', 'ogbn-arxiv', 'ogbn-products', 
                                 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'snap-patents'],
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--split_id', type=int, default=0,
                        help='Split index for datasets providing multiple default splits (e.g., roman-empire)')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID. Use -1 for CPU')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimensionality')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--aggregator', type=str, default='mean',
                        choices=['mean', 'gcn', 'pool', 'lstm'], help='Aggregator type')
    parser.add_argument('--model', type=str, default='sage',
                        choices=['sage', 'gcn', 'gat'], help='Model architecture to use')
    parser.add_argument('--inductive', action='store_true', help='Use inductive training mode (default is transductive)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--fan_out', type=str, default='15,10,5', help='Fan out per layer (comma separated)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    args = parser.parse_args()

    print("=" * 40)
    print("Training Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 40)

    device = torch.device(f'cuda:{args.device}' if args.device >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f"Training mode: {'Inductive' if args.inductive else 'Transductive'}")
    print(f"Model: {args.model.upper()}")

    # Load dataset
    g, features, labels, num_classes, train_idx, val_idx, test_idx = load_dataset(
        args.dataset, args.dataset_dir, split_id=args.split_id
    )
    in_feats = features.shape[1]
    
    if args.inductive:
        # Create training subgraph on CPU to save memory before moving to GPU
        g_train = dgl.node_subgraph(g, train_idx)
        features_train = features[train_idx]
        labels_train = labels[train_idx]
        
        g_train = g_train.to(device)
        features_train = features_train.to(device)
        labels_train = labels_train.to(device)

    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    print(f"Graph Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    print(f"Node feature dim: {in_feats}, Classes: {num_classes}")
    print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # Initialize model
    if args.model == 'sage':
        model = GraphSAGE(in_feats, args.hidden_dim, num_classes, args.n_layers, args.dropout, args.aggregator)
    elif args.model == 'gcn':
        model = GCN(in_feats, args.hidden_dim, num_classes, args.n_layers, args.dropout)
    elif args.model == 'gat':
        # Defaulting num_heads to 4 for GAT, but could be exposed as an arg in the future
        model = GAT(in_feats, args.hidden_dim, num_classes, args.n_layers, args.dropout, num_heads=4)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup DataLoader or Full Graph setup
    fan_out_str = args.fan_out.strip()
    fan_out = [int(fanout) for fanout in fan_out_str.split(',')]
    
    use_full_graph = (len(fan_out) == 1 and fan_out[0] == -1)
    print(f"Using full graph training: {use_full_graph}")
    
    if not use_full_graph:
        if len(fan_out) != args.n_layers:
            print(f"Warning: fan_out length {len(fan_out)} does not match n_layers {args.n_layers}. Adjusting...")
            if len(fan_out) < args.n_layers:
                fan_out = fan_out + [fan_out[-1]] * (args.n_layers - len(fan_out))
            else:
                fan_out = fan_out[:args.n_layers]
                
        sampler = dgl.dataloading.NeighborSampler(fan_out)
        
        if args.inductive:
            # Inductive mode: sample only from the train subgraph
            train_dataloader = dgl.dataloading.DataLoader(
                g_train,
                torch.arange(g_train.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers
            )
        else:
            # Transductive mode: sample from the full graph, but only using train indices as seeds
            train_dataloader = dgl.dataloading.DataLoader(
                g,
                train_idx,
                sampler,
                device=device,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers
            )

    best_val_acc = 0.0
    best_test_acc = 0.0
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0
        num_batches = 0
        
        if use_full_graph:
            if args.inductive:
                logits = model(g_train, features_train)
                loss = F.cross_entropy(logits, labels_train)
            else:
                # Transductive full graph
                logits = model(g, features)
                # calculate loss only on the training nodes
                loss = F.cross_entropy(logits[train_idx], labels[train_idx])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss = loss.item()
        
        else:
            for input_nodes, output_nodes, blocks in train_dataloader:
                if args.inductive:
                    batch_features = features_train[input_nodes]
                    batch_labels = labels_train[output_nodes]
                else:
                    batch_features = features[input_nodes]
                    batch_labels = labels[output_nodes]

                logits = model(blocks, batch_features)
                loss = F.cross_entropy(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches

        train_acc = evaluate(model, g, features, labels, train_idx)
        val_acc = evaluate(model, g, features, labels, val_idx)
        test_acc = evaluate(model, g, features, labels, test_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {time.time() - t0:.4f}s")
              
    print(f"\nOptimization Finished!")
    print(f"Highest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        alloc_mib = peak_alloc / (1024 ** 2)
        reserved_mib = peak_reserved / (1024 ** 2)
        print(f"Peak GPU memory (MiB): allocated={alloc_mib:.2f}, reserved={reserved_mib:.2f}")


if __name__ == '__main__':
    main()
