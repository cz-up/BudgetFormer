import json
import os
import pickle

import torch


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError as exc:
        message = str(exc)
        if "Weights only load failed" in message:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")


def _as_long_idx_tensor(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.tensor(value)
    tensor = tensor.view(-1)
    if tensor.dtype == torch.bool:
        tensor = torch.nonzero(tensor, as_tuple=False).view(-1)
    return tensor.to(torch.long)


def _mask_to_idx(mask: torch.Tensor, split_id: int = 0) -> torch.Tensor:
    mask = mask.detach().cpu()
    if mask.dim() == 2:
        if split_id < 0 or split_id >= mask.size(1):
            raise ValueError(f"split_id={split_id} out of range for mask with {mask.size(1)} splits")
        mask = mask[:, split_id]
    return torch.nonzero(mask.to(torch.bool), as_tuple=False).view(-1).to(torch.long)


def _normalize_split_dict(split_obj):
    if not isinstance(split_obj, dict):
        return None
    train = split_obj.get("train")
    valid = split_obj.get("valid", split_obj.get("val"))
    test = split_obj.get("test")
    if train is None or valid is None or test is None:
        return None
    return {
        "train": _as_long_idx_tensor(train),
        "valid": _as_long_idx_tensor(valid),
        "test": _as_long_idx_tensor(test),
    }


def _load_split_from_dir(dataset_path: str):
    for fname in ("split_idx.pt", "split.pt"):
        fpath = os.path.join(dataset_path, fname)
        if os.path.exists(fpath):
            split_obj = _safe_torch_load(fpath)
            normalized = _normalize_split_dict(split_obj)
            if normalized is not None:
                return normalized

    idx_patterns = [
        ("train_idx.pt", "valid_idx.pt", "test_idx.pt"),
        ("train_idx.pt", "val_idx.pt", "test_idx.pt"),
        ("split/train.pt", "split/valid.pt", "split/test.pt"),
        ("split/train_idx.pt", "split/valid_idx.pt", "split/test_idx.pt"),
    ]
    for tr_name, va_name, te_name in idx_patterns:
        tr_path = os.path.join(dataset_path, tr_name)
        va_path = os.path.join(dataset_path, va_name)
        te_path = os.path.join(dataset_path, te_name)
        if os.path.exists(tr_path) and os.path.exists(va_path) and os.path.exists(te_path):
            return {
                "train": _as_long_idx_tensor(_safe_torch_load(tr_path)),
                "valid": _as_long_idx_tensor(_safe_torch_load(va_path)),
                "test": _as_long_idx_tensor(_safe_torch_load(te_path)),
            }

    mask_patterns = [
        ("train_mask.pt", "valid_mask.pt", "test_mask.pt"),
        ("train_mask.pt", "val_mask.pt", "test_mask.pt"),
        ("split/train_mask.pt", "split/valid_mask.pt", "split/test_mask.pt"),
    ]
    for tr_name, va_name, te_name in mask_patterns:
        tr_path = os.path.join(dataset_path, tr_name)
        va_path = os.path.join(dataset_path, va_name)
        te_path = os.path.join(dataset_path, te_name)
        if os.path.exists(tr_path) and os.path.exists(va_path) and os.path.exists(te_path):
            return {
                "train": _mask_to_idx(_safe_torch_load(tr_path)),
                "valid": _mask_to_idx(_safe_torch_load(va_path)),
                "test": _mask_to_idx(_safe_torch_load(te_path)),
            }

    role_path = os.path.join(dataset_path, "role.json")
    if os.path.exists(role_path):
        with open(role_path, "r", encoding="utf-8") as handle:
            role = json.load(handle)
        train = role.get("tr", role.get("train"))
        valid = role.get("va", role.get("valid", role.get("val")))
        test = role.get("te", role.get("test"))
        if train is not None and valid is not None and test is not None:
            return {
                "train": _as_long_idx_tensor(train),
                "valid": _as_long_idx_tensor(valid),
                "test": _as_long_idx_tensor(test),
            }

    return None


def _load_pyg_default_split(dataset_name: str, root_dir: str):
    name = str(dataset_name)
    key = name.lower()
    data = None

    if key in {"cora", "citeseer", "pubmed"}:
        from torch_geometric.datasets import Planetoid

        data = Planetoid(root=root_dir, name=key)[0]
    elif key in {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}:
        from torch_geometric.datasets import HeterophilousGraphDataset

        data = HeterophilousGraphDataset(root=root_dir, name=name.capitalize())[0]
    elif key in {"photo", "computers", "amazon-photo", "amazon-computers"}:
        from torch_geometric.datasets import Amazon

        pyg_name = "Photo" if "photo" in key else "Computers"
        data = Amazon(root=root_dir, name=pyg_name)[0]

    if data is None:
        return None
    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask") or not hasattr(data, "test_mask"):
        return None
    return {
        "train": _mask_to_idx(data.train_mask),
        "valid": _mask_to_idx(data.val_mask),
        "test": _mask_to_idx(data.test_mask),
    }


def load_default_split(dataset_name: str, root_dir: str, dist_module=None, wait_for_rank0: bool = False):
    dataset_path = os.path.join(root_dir, dataset_name)
    should_wait = wait_for_rank0 and dist_module is not None and dist_module.is_initialized()
    is_rank0 = True if not should_wait else (dist_module.get_rank() == 0)

    if should_wait and not is_rank0:
        dist_module.barrier()

    split_idx = _load_split_from_dir(dataset_path)
    if split_idx is not None:
        if should_wait and is_rank0:
            dist_module.barrier()
        return split_idx

    name = str(dataset_name)
    if name.startswith("ogbn-"):
        try:
            from ogb.nodeproppred import NodePropPredDataset
        except Exception:
            if should_wait and is_rank0:
                dist_module.barrier()
            return None
        split_obj = NodePropPredDataset(name=name, root=root_dir).get_idx_split()
        split_idx = _normalize_split_dict(split_obj)
        if should_wait and is_rank0:
            dist_module.barrier()
        return split_idx

    try:
        split_idx = _load_pyg_default_split(name, root_dir)
    except Exception:
        split_idx = None

    if should_wait and is_rank0:
        dist_module.barrier()
    return split_idx
