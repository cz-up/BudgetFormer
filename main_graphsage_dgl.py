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
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            if i != self.n_layers - 1:
                h = F.relu(h)
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

    elif dataset_name == 'roman-empire':
        from torch_geometric.datasets import HeterophilousGraphDataset
        pyg_data = HeterophilousGraphDataset(root=root, name="Roman-empire")[0]

        src = pyg_data.edge_index[0].to(torch.long)
        dst = pyg_data.edge_index[1].to(torch.long)
        num_nodes = int(pyg_data.x.size(0))
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        features = pyg_data.x.to(torch.float32)
        labels = pyg_data.y.to(torch.long).view(-1)
        num_classes = int(labels.max().item()) + 1

        if not hasattr(pyg_data, "train_mask") or not hasattr(pyg_data, "val_mask") or not hasattr(pyg_data, "test_mask"):
            raise ValueError("Roman-empire dataset does not provide default train/val/test masks.")
        train_idx = _mask_to_index(pyg_data.train_mask, split_id=split_id)
        val_idx = _mask_to_index(pyg_data.val_mask, split_id=split_id)
        test_idx = _mask_to_index(pyg_data.test_mask, split_id=split_id)
        
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
    parser = argparse.ArgumentParser(description='GraphSAGE Node Classification using DGL')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['cora', 'citeseer', 'pubmed', 'reddit', 'ogbn-arxiv', 'ogbn-products', 'roman-empire'],
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--split_id', type=int, default=0,
                        help='Split index for datasets providing multiple default splits (e.g., roman-empire)')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID. Use -1 for CPU')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GraphSAGE layers')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimensionality')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--aggregator', type=str, default='mean',
                        choices=['mean', 'gcn', 'pool', 'lstm'], help='Aggregator type')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if args.device >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load dataset
    g, features, labels, num_classes, train_idx, val_idx, test_idx = load_dataset(
        args.dataset, args.dataset_dir, split_id=args.split_id
    )
    in_feats = features.shape[1]
    
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
    model = GraphSAGE(in_feats, args.hidden_dim, num_classes, args.n_layers, args.dropout, args.aggregator)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = evaluate(model, g, features, labels, train_idx)
        val_acc = evaluate(model, g, features, labels, val_idx)
        test_acc = evaluate(model, g, features, labels, test_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {time.time() - t0:.4f}s")
              
    print(f"\nOptimization Finished!")
    print(f"Highest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")


if __name__ == '__main__':
    main()
