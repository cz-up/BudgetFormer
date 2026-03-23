import argparse
import json
import os

"""
usage:
    single dataset, multiple epochs:
        python3 plot/attn_trend.py \
        --stats_root ./plot/hop_stats \
        --model graphormer \
        --dataset ogbn-products \
        --epochs 10,20,30 \
        --layer 0 \
        --mass 1 \
        --max_hop 0 \
        --seq_len 8000 \
        --out plot/products_layer0_epochs.png 
    multiple datasets, same epoch:
        python3 plot/attn_trend.py \
        --stats_root ./plot/hop_stats \
        --model graphormer \
        --datasets cora,citeseer,pubmed,ogbn-arxiv,ogbn-products \
        --epoch 30 \
        --layer 0 \
        --mass 1 \
        --max_hop 0 \
        --seq_len 8000 \
        --out plot/layer0_multi_dataset_epoch30.png
"""

def load_legacy_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    data = {}
    for ds, layers in raw.items():
        data[ds] = {int(k): v for k, v in layers.items()}
    return data


def _candidate_stats_dirnames(mass, tag="", max_hop=None, seq_len=None):
    base = f"attn{mass:g}"
    tag = str(tag).strip()
    names = []
    if max_hop is not None and seq_len is not None:
        dirname = f"{base}_maxhop{int(max_hop)}_seq{int(seq_len)}"
        if tag:
            dirname = f"{dirname}_{tag}"
        names.append(dirname)
    legacy = base
    if tag:
        legacy = f"{legacy}_{tag}"
    names.append(legacy)
    return names


def _resolve_stats_dir(root_dir, model, dataset, mass, tag="", max_hop=None, seq_len=None):
    dataset_root = os.path.join(root_dir, model, dataset)
    if not os.path.isdir(dataset_root):
        return None

    for dirname in _candidate_stats_dirnames(mass, tag=tag, max_hop=max_hop, seq_len=seq_len):
        stats_dir = os.path.join(dataset_root, dirname)
        if os.path.isdir(stats_dir):
            return stats_dir

    base = f"attn{mass:g}"
    tag = str(tag).strip()
    candidates = []
    for name in sorted(os.listdir(dataset_root)):
        full_path = os.path.join(dataset_root, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith(base):
            continue
        if tag and not name.endswith(f"_{tag}"):
            continue
        if max_hop is not None and f"_maxhop{int(max_hop)}_" not in name:
            continue
        if seq_len is not None and f"_seq{int(seq_len)}" not in name:
            continue
        candidates.append(full_path)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _load_epoch_record(stats_dir, epoch=None):
    if epoch is None:
        path = os.path.join(stats_dir, "latest.json")
        if not os.path.exists(path):
            return None
    else:
        path = os.path.join(stats_dir, f"epoch_{int(epoch):04d}.json")
        if not os.path.exists(path):
            return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stats_dir(root_dir, model, mass, epoch=None, datasets=None, tag="", max_hop=None, seq_len=None):
    model_root = os.path.join(root_dir, model)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"Stats root not found for model {model!r}: {model_root}")

    if datasets:
        dataset_names = [ds.strip() for ds in datasets.split(",") if ds.strip()]
    else:
        dataset_names = sorted(
            name for name in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, name))
        )

    out = {}
    for dataset in dataset_names:
        stats_dir = _resolve_stats_dir(
            root_dir,
            model,
            dataset,
            mass,
            tag=tag,
            max_hop=max_hop,
            seq_len=seq_len,
        )
        if not stats_dir or not os.path.isdir(stats_dir):
            continue
        record = _load_epoch_record(stats_dir, epoch=epoch)
        if record is None:
            continue
        layers = {}
        for layer, item in record.get("layers", {}).items():
            layers[int(layer)] = item.get("hop_ratios", [])
        if layers:
            out[dataset] = layers
    return out


def load_dataset_epochs(root_dir, model, dataset, mass, epochs, tag="", max_hop=None, seq_len=None):
    stats_dir = _resolve_stats_dir(
        root_dir,
        model,
        dataset,
        mass,
        tag=tag,
        max_hop=max_hop,
        seq_len=seq_len,
    )
    if not stats_dir or not os.path.isdir(stats_dir):
        return {}

    out = {}
    for epoch in epochs:
        record = _load_epoch_record(stats_dir, epoch=epoch)
        if record is None:
            continue
        layers = {}
        for layer, item in record.get("layers", {}).items():
            layers[int(layer)] = item.get("hop_ratios", [])
        if layers:
            out[int(epoch)] = layers
    return out


def plot_hop_ratios(data, use_log=False, use_symlog=False, out_path="hop_ratios.png", layer=None, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.figure(figsize=(7, 4))

    if len(data) > 1:
        target_layer = 0 if layer is None else int(layer)
        for ds_name, layers in data.items():
            if target_layer not in layers:
                continue
            ratios = layers[target_layer]
            hops = np.arange(1, len(ratios) + 1)
            plt.plot(hops, ratios, marker="o", label=f"{ds_name}")
        if title is None:
            title = f"Attention hop ratios - layer {target_layer} across datasets"
    else:
        ds_name = next(iter(data))
        layers = data[ds_name]
        for layer_id, ratios in sorted(layers.items()):
            hops = np.arange(1, len(ratios) + 1)
            plt.plot(hops, ratios, marker="o", label=f"layer {layer_id}")
        if title is None:
            title = f"Attention hop ratios - {ds_name}"

    plt.title(title)
    plt.xlabel("Hop distance")
    plt.ylabel("Attention ratio")
    if use_symlog:
        plt.yscale("symlog", linthresh=1e-3)
    elif use_log:
        plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_epoch_hop_ratios(data, dataset, layer, use_log=False, use_symlog=False, out_path="hop_ratios_epochs.png", title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    target_layer = int(layer)
    plt.figure(figsize=(7, 4))

    for epoch in sorted(data):
        layers = data[epoch]
        if target_layer not in layers:
            continue
        ratios = layers[target_layer]
        hops = np.arange(1, len(ratios) + 1)
        plt.plot(hops, ratios, marker="o", label=f"epoch {epoch}")

    if title is None:
        title = f"Attention hop ratios - {dataset} layer {target_layer} across epochs"

    plt.title(title)
    plt.xlabel("Hop distance")
    plt.ylabel("Attention ratio")
    if use_symlog:
        plt.yscale("symlog", linthresh=1e-3)
    elif use_log:
        plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot attention hop-ratio curves from saved JSON stats.")
    parser.add_argument("--input_json", type=str, default="", help="legacy aggregated JSON path; if set, stats directory scanning is skipped")
    parser.add_argument("--stats_root", type=str, default="./hop_stats", help="root directory for saved per-epoch hop-mass stats")
    parser.add_argument("--model", type=str, default="graphormer", help="model name under stats_root")
    parser.add_argument("--mass", type=float, default=0.95, help="hop-mass target used in the saved stats")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to read; <=0 uses latest.json")
    parser.add_argument("--dataset", type=str, default="", help="single dataset name for multi-epoch plotting")
    parser.add_argument("--epochs", type=str, default="", help="comma-separated epoch list for single-dataset multi-epoch plotting")
    parser.add_argument("--datasets", type=str, default="", help="optional comma-separated dataset list")
    parser.add_argument("--tag", type=str, default="", help="optional stats directory tag suffix")
    parser.add_argument("--max_hop", type=int, default=-1, help="optional max-hop directory component; <0 disables the filter")
    parser.add_argument("--seq_len", type=int, default=-1, help="optional seq-len directory component; <0 disables the filter")
    parser.add_argument("--layer", type=int, default=0, help="layer to compare across datasets when plotting multiple datasets")
    parser.add_argument("--use_log", action="store_true", default=True, help="use log y-scale")
    parser.add_argument("--no_log", action="store_false", dest="use_log", help="disable log y-scale")
    parser.add_argument("--use_symlog", action="store_true", default=False, help="use symlog y-scale")
    parser.add_argument("--out", type=str, default="hop_ratios.png", help="output image path")
    args = parser.parse_args()

    epoch_list = [int(item.strip()) for item in args.epochs.split(",") if item.strip()]

    if args.input_json:
        data = load_legacy_json(args.input_json)
        mode = "legacy"
    elif args.dataset and epoch_list:
        data = load_dataset_epochs(
            root_dir=args.stats_root,
            model=args.model,
            dataset=args.dataset,
            mass=args.mass,
            epochs=epoch_list,
            tag=args.tag,
            max_hop=(None if args.max_hop < 0 else args.max_hop),
            seq_len=(None if args.seq_len < 0 else args.seq_len),
        )
        mode = "dataset_epochs"
    else:
        data = load_stats_dir(
            root_dir=args.stats_root,
            model=args.model,
            mass=args.mass,
            epoch=(None if args.epoch <= 0 else args.epoch),
            datasets=args.datasets,
            tag=args.tag,
            max_hop=(None if args.max_hop < 0 else args.max_hop),
            seq_len=(None if args.seq_len < 0 else args.seq_len),
        )
        mode = "default"

    if not data:
        raise ValueError("No hop-ratio data found for the requested settings.")

    if mode == "dataset_epochs":
        plot_epoch_hop_ratios(
            data,
            dataset=args.dataset,
            layer=args.layer,
            use_log=args.use_log,
            use_symlog=args.use_symlog,
            out_path=args.out,
        )
    else:
        plot_hop_ratios(
            data,
            use_log=args.use_log,
            use_symlog=args.use_symlog,
            out_path=args.out,
            layer=args.layer,
        )


if __name__ == "__main__":
    main()
