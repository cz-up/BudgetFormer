import argparse
import time
import torch as th
from dgl.nn.pytorch.gt import GraphormerLayer
from utils.lr import PolynomialDecayLR


def make_padding_mask(num_nodes, nmax, device):
    # True 表示要 mask
    bsz = num_nodes.shape[0]
    idx = th.arange(nmax, device=device)[None, :]  # (1, N)
    valid = idx < num_nodes[:, None]              # (B, N)
    attn_mask = (~valid)[:, None, :].expand(bsz, nmax, nmax).clone()
    pad_queries = (~valid)[:, :, None].expand(bsz, nmax, nmax)
    attn_mask[pad_queries] = False
    return attn_mask


def pad_x(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x


def pad_y(y, padlen):
    ylen = y.size(0)
    if ylen < padlen:
        new_y = th.full((padlen,), -100, dtype=y.dtype, device=y.device)
        new_y[:ylen] = y
        y = new_y
    return y


def random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=42):
    rng = th.Generator()
    rng.manual_seed(seed)
    n = y.size(0)
    perm = th.randperm(n, generator=rng)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:n_train + n_valid]
    test_idx = perm[n_train + n_valid:]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def load_ogbn_split(dataset_name, root_dir):
    name = dataset_name[:-2] if dataset_name.endswith("_n") else dataset_name
    if not name.startswith("ogbn-"):
        return None
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except Exception:
        return None
    dataset = NodePropPredDataset(name=name, root=root_dir)
    split_idx = dataset.get_idx_split()
    out = {}
    for key, val in split_idx.items():
        if isinstance(val, th.Tensor):
            out[key] = val.to(th.long)
        else:
            out[key] = th.tensor(val, dtype=th.long)
    return out


class DGLGraphormer(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ffn_dim, num_classes, dropout, attn_dropout):
        super().__init__()
        self.input_proj = th.nn.Linear(input_dim, hidden_dim)
        self.layers = th.nn.ModuleList([
            GraphormerLayer(hidden_dim, ffn_dim, num_heads, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = th.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, attn_mask=None):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, attn_bias=None, attn_mask=attn_mask)
        return th.log_softmax(self.out_proj(x), dim=-1)


def run_epoch(model, optimizer, x, y, idx, seq_len, device, train=True,
              transductive=False, all_idx=None):
    """
    Args:
        idx      : the split indices whose loss/accuracy should be computed
                   (train / valid / test).
        transductive: if True, each batch is built from *all* nodes (all_idx)
                   so every node participates in self-attention; loss and
                   accuracy are still computed only on nodes in `idx`.
        all_idx  : full node index pool used as the sequence source when
                   transductive=True.  Ignored when transductive=False.
    """
    if transductive and all_idx is not None:
        # Shuffle for training so context nodes are randomly mixed;
        # for eval keep a deterministic order (avoids re-shuffling noise).
        pool = th.randperm(all_idx.size(0)) if train else th.arange(all_idx.size(0))
        pool = all_idx[pool]
        idx_set = set(idx.tolist())
    else:
        pool = idx
        idx_set = None  # not needed; every node in pool is a target node

    num_batch = pool.size(0) // seq_len + 1
    total_loss = 0.0
    correct = 0
    total = 0
    for i in range(num_batch):
        batch_nodes = pool[i * seq_len:(i + 1) * seq_len]
        if batch_nodes.numel() == 0:
            continue
        x_i = x[batch_nodes].to(device)
        y_i = y[batch_nodes].to(device)

        # Build padding mask when this slice is smaller than seq_len
        actual = batch_nodes.numel()
        if actual < seq_len:
            x_i = pad_x(x_i, seq_len)
            y_i = pad_y(y_i, seq_len)
            num_nodes = th.tensor([actual], device=device)
            attn_mask = make_padding_mask(num_nodes, seq_len, device)
        else:
            attn_mask = None

        x_i = x_i.unsqueeze(0)  # [1, N, D]

        if transductive and idx_set is not None:
            # Build a boolean mask: which positions in this batch are
            # target nodes (belong to `idx`)?
            loss_mask = th.tensor(
                [n.item() in idx_set for n in batch_nodes],
                dtype=th.bool, device=device
            )  # length == actual (un-padded)
            # Extend mask to seq_len if padded
            if actual < seq_len:
                pad_extra = th.zeros(seq_len - actual, dtype=th.bool, device=device)
                loss_mask = th.cat([loss_mask, pad_extra])
        else:
            loss_mask = None  # use all valid (non-padded) nodes

        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_i, attn_mask=attn_mask)[0]
            if loss_mask is not None and loss_mask.any():
                loss = th.nn.functional.nll_loss(logits[loss_mask], y_i[loss_mask])
            elif loss_mask is None:
                loss = th.nn.functional.nll_loss(logits, y_i)
            else:
                continue  # no target nodes in this batch
            loss.backward()
            optimizer.step()
        else:
            with th.no_grad():
                logits = model(x_i, attn_mask=attn_mask)[0]
                if loss_mask is not None and loss_mask.any():
                    loss = th.nn.functional.nll_loss(logits[loss_mask], y_i[loss_mask])
                elif loss_mask is None:
                    loss = th.nn.functional.nll_loss(logits, y_i)
                else:
                    continue

        total_loss += float(loss.item())
        # Accuracy: only over target positions that have a valid label
        if loss_mask is not None:
            valid = (y_i >= 0) & loss_mask
        else:
            valid = y_i >= 0
        pred = logits.argmax(dim=-1)
        correct += int((pred[valid] == y_i[valid]).sum().item())
        total += int(valid.sum().item())
    acc = correct / max(1, total)
    return total_loss / max(1, num_batch), acc


def main():
    parser = argparse.ArgumentParser(description="DGL GraphormerLayer node-level training")
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/")
    parser.add_argument("--seq_len", type=int, default=8000)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--peak_lr", type=float, default=2e-4)
    parser.add_argument("--end_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_updates", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--attention_dropout_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ogbn_split", type=int, default=0)
    parser.add_argument("--transductive", type=int, default=0,
                        help="1 = transductive: all nodes see each other in attention; "
                             "0 = inductive (default): each batch contains only the target split nodes")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not th.cuda.is_available():
        device = "cpu"

    x = th.load(args.dataset_dir + args.dataset + "/x.pt")
    y = th.load(args.dataset_dir + args.dataset + "/y.pt")
    num_classes = int(y.max().item()) + 1

    if args.use_ogbn_split:
        split_idx = load_ogbn_split(args.dataset, args.dataset_dir)
        if split_idx is None:
            print("Warning: ogbn split unavailable, fallback to random split.")
            split_idx = random_split_idx(y, seed=args.seed)
    else:
        split_idx = random_split_idx(y, seed=args.seed)

    model = DGLGraphormer(
        input_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        num_classes=num_classes,
        dropout=args.dropout_rate,
        attn_dropout=args.attention_dropout_rate,
    ).to(device)

    optimizer = th.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_updates,
        tot=args.epochs,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    best_epoch = -1
    best_val = 0.0
    best_test = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        transductive = bool(args.transductive)
        all_idx = th.arange(x.size(0)) if transductive else None
        train_loss, train_acc = run_epoch(
            model, optimizer, x, y, split_idx["train"], args.seq_len, device, train=True,
            transductive=transductive, all_idx=all_idx
        )
        lr_scheduler.step()
        model.eval()
        val_loss, val_acc = run_epoch(
            model, optimizer, x, y, split_idx["valid"], args.seq_len, device, train=False,
            transductive=transductive, all_idx=all_idx
        )
        test_loss, test_acc = run_epoch(
            model, optimizer, x, y, split_idx["test"], args.seq_len, device, train=False,
            transductive=transductive, all_idx=all_idx
        )
        print(
            f"Epoch {epoch:03d} "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} "
            f"val loss {val_loss:.4f} acc {val_acc:.2%} "
            f"test loss {test_loss:.4f} acc {test_acc:.2%}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
        if test_acc > best_test:
            best_test = test_acc

    print(f"Best epoch: {best_epoch}, validation accuracy: {best_val:.2%}, test accuracy: {best_test:.2%}")
    if th.cuda.is_available():
        peak_alloc = th.cuda.max_memory_allocated()
        peak_reserved = th.cuda.max_memory_reserved()
        alloc_mib = peak_alloc / (1024 ** 2)
        reserved_mib = peak_reserved / (1024 ** 2)
        print(f"Peak GPU memory (MiB): allocated={alloc_mib:.2f}, reserved={reserved_mib:.2f}")


if __name__ == "__main__":
    main()
