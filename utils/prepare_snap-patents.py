"""Prepare snap-patents to match the official LINKX benchmark format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

_LINKX_SNAP_PATENTS_SPLITS_DRIVE_ID = "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N"


def even_quantile_labels(vals: np.ndarray, nclasses: int = 5, verbose: bool = True) -> np.ndarray:
    """Official LINKX label conversion for snap-patents."""
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) & (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"  class {class_idx}: [{interval[0]}, {interval[1]})")
    return label


def _as_long_idx_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    return tensor.view(-1).to(torch.long)


def _normalize_split_dict(split_obj: dict) -> dict[str, torch.Tensor]:
    train = split_obj.get("train")
    valid = split_obj.get("valid", split_obj.get("val"))
    test = split_obj.get("test")
    if train is None or valid is None or test is None:
        raise ValueError("Split object is missing train/valid/test keys.")
    return {
        "train": _as_long_idx_tensor(train),
        "valid": _as_long_idx_tensor(valid),
        "test": _as_long_idx_tensor(test),
    }


def _load_raw_year_labels(dataset_path: Path, label_filename: str) -> tuple[torch.Tensor, Path]:
    raw_year_path = dataset_path / "y_raw_year.pt"
    if raw_year_path.exists():
        years = torch.load(raw_year_path, map_location="cpu").view(-1).to(torch.long)
        return years, raw_year_path

    label_path = dataset_path / label_filename
    if not label_path.exists():
        raise FileNotFoundError(f"Raw label file not found: {label_path}")

    labels = torch.load(label_path, map_location="cpu").view(-1).to(torch.long)
    return labels, label_path


def prepare_labels(dataset_path: Path, label_filename: str, nclasses: int, force: bool) -> None:
    y_path = dataset_path / label_filename
    years, source_path = _load_raw_year_labels(dataset_path, label_filename)

    if years.unique().numel() <= nclasses and years.min().item() >= 0 and years.max().item() < nclasses and not force:
        print(f"[labels] {y_path} already looks like {nclasses}-class labels; skipping conversion.")
        return

    raw_year_path = dataset_path / "y_raw_year.pt"
    if source_path != raw_year_path:
        torch.save(years.clone(), raw_year_path)
        print(f"[labels] Backed up raw year labels to {raw_year_path}")

    y_quantized = torch.from_numpy(
        even_quantile_labels(years.cpu().numpy(), nclasses=nclasses, verbose=True)
    ).to(torch.long)
    torch.save(y_quantized, y_path)
    print(f"[labels] Saved LINKX-style {nclasses}-class labels to {y_path}")

    vals, cnts = torch.unique(y_quantized, return_counts=True)
    print(f"[labels] num_unique={vals.numel()} classes={vals.tolist()}")
    print("[labels] class counts:")
    for v, c in zip(vals.tolist(), cnts.tolist()):
        print(f"  {v}: {c}")


def download_official_splits(out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"[splits] Reusing existing file: {out_path}")
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required to download the official LINKX split file.") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[splits] Downloading official snap-patents splits to {out_path}")
    drive_url = f"https://drive.google.com/uc?id={_LINKX_SNAP_PATENTS_SPLITS_DRIVE_ID}"
    try:
        downloaded = gdown.download(
            url=drive_url,
            output=str(out_path),
            quiet=False,
            fuzzy=True,
        )
    except TypeError:
        downloaded = gdown.download(
            url=drive_url,
            output=str(out_path),
            quiet=False,
        )
    if downloaded is None or not out_path.exists():
        raise RuntimeError("Failed to download official snap-patents splits.")


def prepare_splits(dataset_path: Path, split_id: int, force: bool) -> None:
    npy_path = dataset_path / "official_splits.npy"
    download_official_splits(npy_path, force=force)

    splits_raw = np.load(npy_path, allow_pickle=True)
    if isinstance(splits_raw, np.ndarray) and splits_raw.dtype == object:
        splits_seq = splits_raw.tolist()
    else:
        splits_seq = list(splits_raw)
    if not splits_seq:
        raise RuntimeError("Official split file is empty.")
    if split_id < 0 or split_id >= len(splits_seq):
        raise ValueError(f"split_id={split_id} out of range for {len(splits_seq)} official splits")

    split_list = [_normalize_split_dict(split_obj) for split_obj in splits_seq]

    all_pt_path = dataset_path / "split_idx_all.pt"
    one_pt_path = dataset_path / "split_idx.pt"
    torch.save(split_list, all_pt_path)
    torch.save(split_list[split_id], one_pt_path)
    print(f"[splits] Saved all {len(split_list)} official splits to {all_pt_path}")
    print(f"[splits] Saved split_id={split_id} as default split file to {one_pt_path}")
    for split_name, idx in split_list[split_id].items():
        print(f"  {split_name}: {int(idx.numel()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare snap-patents using the official LINKX label/split conventions.")
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--dataset", type=str, default="snap-patents")
    parser.add_argument("--label_filename", type=str, default="y.pt")
    parser.add_argument("--nclasses", type=int, default=5)
    parser.add_argument("--split_id", type=int, default=0, help="which official split to materialize as split_idx.pt")
    parser.add_argument("--splits_only", action="store_true", help="download/convert splits without touching labels")
    parser.add_argument("--labels_only", action="store_true", help="convert labels without downloading splits")
    parser.add_argument("--force", action="store_true", help="overwrite cached outputs and reconvert labels")
    args = parser.parse_args()

    if args.splits_only and args.labels_only:
        raise ValueError("--splits_only and --labels_only are mutually exclusive.")

    dataset_path = Path(args.dataset_dir) / args.dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    if not args.splits_only:
        prepare_labels(dataset_path, args.label_filename, nclasses=int(args.nclasses), force=bool(args.force))
    if not args.labels_only:
        prepare_splits(dataset_path, split_id=int(args.split_id), force=bool(args.force))


if __name__ == "__main__":
    main()
