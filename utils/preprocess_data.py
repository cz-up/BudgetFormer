import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
import scipy
import scipy.io
from numpy.linalg import inv
from torch_geometric.datasets import Planetoid, Amazon, Actor, CitationFull, Coauthor, HeterophilousGraphDataset
from torch.nn.functional import normalize
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce
from tqdm import tqdm
import os
import json
import shutil
import random
import math
import pickle as pkl
from ogb.nodeproppred import NodePropPredDataset
import argparse

def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def _load_linkx_mat(mat_path: str, label_key: str):
    """Load a LINKX-format .mat file (edge_index, node_feat, <label_key>, num_nodes)."""
    fulldata = scipy.io.loadmat(mat_path)

    edge_index = torch.as_tensor(fulldata['edge_index'], dtype=torch.long)

    node_feat = fulldata['node_feat']
    if sp.issparse(node_feat):
        node_feat = node_feat.toarray()
    data_x = torch.as_tensor(np.asarray(node_feat), dtype=torch.float32)

    label = np.asarray(fulldata[label_key]).reshape(-1)
    data_y = torch.as_tensor(label, dtype=torch.long)

    return data_x, edge_index, data_y


def _load_snap_patents_mat(mat_path: str):
    """Load the official LINKX snap-patents .mat file.

    Expects the keys used by the LINKX release: edge_index, node_feat
    (sparse), years, num_nodes. Returns raw (unquantized) year labels;
    prepare_snap-patents.py handles the quantile->class conversion.
    """
    return _load_linkx_mat(mat_path, label_key='years')


def rand_train_test_idx(label: torch.Tensor, train_prop: float, valid_prop: float, rng: np.random.RandomState,
                         ignore_negative: bool = True) -> dict:
    """Official LINKX random label split, used for datasets with no fixed official split (e.g. genius)."""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = torch.arange(label.shape[0])

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(rng.permutation(n))

    train_idx = labeled_nodes[perm[:train_num]]
    valid_idx = labeled_nodes[perm[train_num:train_num + valid_num]]
    test_idx = labeled_nodes[perm[train_num + valid_num:]]

    return {
        "train": train_idx.to(torch.long),
        "valid": valid_idx.to(torch.long),
        "test": test_idx.to(torch.long),
    }


def _resolve_mat_path(dataset_dir: str, dataset_name: str, mat_filename: str) -> str:
    mat_path = os.path.join(dataset_dir, dataset_name, mat_filename)
    if os.path.exists(mat_path):
        return mat_path

    legacy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', mat_filename)
    if os.path.exists(legacy_path):
        return legacy_path

    raise FileNotFoundError(
        f"{mat_filename} not found at {mat_path} or {legacy_path}. "
        "Download it from the LINKX release and place it in one of these locations."
    )


def _download_gdrive_file(file_id: str, out_path: str, force: bool = False) -> None:
    if os.path.exists(out_path) and not force:
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required to download this file (pip install gdown).") from exc

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    downloaded = gdown.download(id=file_id, output=out_path, quiet=False)
    if downloaded is None or not os.path.exists(out_path):
        raise RuntimeError(f"Failed to download {out_path} from Google Drive (id={file_id}).")


def _resolve_raw_dataset_dir(dataset_dir: str, dataset_name: str, required_files) -> str:
    """Find a directory containing all `required_files`, checking the output
    location first (`{dataset_dir}/{dataset_name}`) and falling back to the
    legacy raw-data location (`utils/dataset/{dataset_name}`, resolved
    relative to this file so it works regardless of the caller's cwd)."""
    candidate = os.path.join(dataset_dir, dataset_name)
    if all(os.path.exists(os.path.join(candidate, fname)) for fname in required_files):
        return candidate

    legacy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', dataset_name)
    if all(os.path.exists(os.path.join(legacy_dir, fname)) for fname in required_files):
        return legacy_dir

    raise FileNotFoundError(
        f"Raw files {list(required_files)} not found under {candidate} or {legacy_dir}."
    )


def _mask_to_index(mask: torch.Tensor, split_id: int = 0) -> torch.Tensor:
    if mask.dim() == 2:
        if split_id < 0 or split_id >= mask.size(1):
            raise ValueError(f"split_id={split_id} out of range for mask with {mask.size(1)} splits")
        mask = mask[:, split_id]
    return torch.nonzero(mask.to(torch.bool), as_tuple=True)[0]


def get_dataset(dataset_name, split_id: int = 0):
    print(f'Get dataset {dataset_name}...')

    dataset_dir = './dataset/'
    if not os.path.exists(f'{dataset_dir}/{dataset_name}'): 
        os.makedirs(f'{dataset_dir}/{dataset_name}')

    if True:
        if dataset_name in ['cora', 'citeseer', 'pubmed']: 
            dataset = Planetoid(root=dataset_dir, name=dataset_name)       
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ['dblp']:
            dataset = CitationFull(root=dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
                
        elif dataset_name in ["CS", "Physics"]:
        # TODO: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Coauthor.html
            dataset = Coauthor(root=dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ["Photo"]:
            dataset = Amazon(root=dataset_dir, name=dataset_name)
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
        
        elif dataset_name in ['aminer']:
            adj = pkl.load(open(os.path.join(dataset_dir, dataset_name, "{}.adj.sp.pkl".format(dataset_name)), "rb"))
            data_x = pkl.load(
                open(os.path.join(dataset_dir, dataset_name, "{}.features.pkl".format(dataset_name)), "rb"))
            data_y = pkl.load(
                open(os.path.join(dataset_dir, dataset_name, "{}.labels.pkl".format(dataset_name)), "rb"))
            # random_state = np.random.RandomState(split_seed)
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['reddit']:
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, '{}_adj.npz'.format(dataset_name)))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, '{}_feat.npy'.format(dataset_name)))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, '{}_labels.npy'.format(dataset_name)))
            # random_state = np.random.RandomState(split_seed)
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))

            
        elif dataset_name in ['Amazon2M']:
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, '{}_adj.npz'.format(dataset_name)))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, '{}_feat.npy'.format(dataset_name)))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, '{}_labels.npy'.format(dataset_name)))
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['amazon']:
            raw_dir = _resolve_raw_dataset_dir(dataset_dir, dataset_name, ('adj_full.npz', 'feats.npy', 'labels.npy'))
            adj = sp.load_npz(os.path.join(raw_dir, 'adj_full.npz'))
            data_x = np.load(os.path.join(raw_dir, 'feats.npy'))
            data_y = np.load(os.path.join(raw_dir, 'labels.npy'))
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))

            # role.json is the official GraphSAINT/PyG AmazonProducts split (fixed
            # 80/5/15 train/valid/test), downloaded once and cached alongside the
            # other raw files so later runs reuse it instead of re-downloading.
            role_path = os.path.join(raw_dir, 'role.json')
            _download_gdrive_file('1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K', role_path)
            with open(role_path, 'r', encoding='utf-8') as f:
                role = json.load(f)
            split_idx = {
                "train": torch.as_tensor(role['tr'], dtype=torch.long),
                "valid": torch.as_tensor(role['va'], dtype=torch.long),
                "test": torch.as_tensor(role['te'], dtype=torch.long),
            }

            # Also mirror role.json into the output dir so it matches the
            # historical file layout (split_utils.py accepts either format).
            output_role_path = os.path.join(dataset_dir, dataset_name, 'role.json')
            if os.path.abspath(output_role_path) != os.path.abspath(role_path):
                shutil.copyfile(role_path, output_role_path)

        elif dataset_name in ['pokec']:
            fulldata = scipy.io.loadmat(f'/path/to/dataset/pokec.mat')
            edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
            
            data_x = torch.tensor(fulldata['node_feat']).float()
            label = fulldata['label'].flatten()
            data_y = torch.tensor(label, dtype=torch.long)
            
            num_nodes = data_y.shape[0]
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                        shape=(num_nodes, num_nodes), dtype=np.float32)
          
            
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ['snap-patents']:
            mat_path = _resolve_mat_path(dataset_dir, dataset_name, 'snap_patents.mat')
            data_x, edge_index, data_y = _load_snap_patents_mat(mat_path)
            # data_y holds raw publication years here; utils/prepare_snap-patents.py
            # converts them to LINKX-style quantile classes afterwards.

        elif dataset_name in ['genius']:
            mat_path = _resolve_mat_path(dataset_dir, dataset_name, 'genius.mat')
            data_x, edge_index, data_y = _load_linkx_mat(mat_path, label_key='label')
            # genius ships final binary labels directly, unlike snap-patents' raw years.

            # genius has no official fixed split; LINKX draws a random 50/25/25 split at
            # runtime. Generate GENIUS_NUM_SPLITS reproducible splits (seeded off
            # GENIUS_SPLIT_SEED + split index) and keep `split_id` as the default split_idx.pt.
            genius_num_splits = 5
            genius_split_seed = 0
            if split_id < 0 or split_id >= genius_num_splits:
                raise ValueError(f"split_id={split_id} out of range for {genius_num_splits} genius splits")
            split_list = [
                rand_train_test_idx(data_y, train_prop=0.5, valid_prop=0.25,
                                     rng=np.random.RandomState(genius_split_seed + i))
                for i in range(genius_num_splits)
            ]
            os.makedirs(os.path.join(dataset_dir, dataset_name), exist_ok=True)
            torch.save(split_list, os.path.join(dataset_dir, dataset_name, 'split_idx_all.pt'))
            split_idx = split_list[split_id]

        elif dataset_name in {"ogbn-papers100M"}:
            file_dir = '/home/zhaochu/torchgt/utils/dataset/'
            ogb_dataset = NodePropPredDataset(name=dataset_name, root=file_dir)
            split_idx = ogb_dataset.get_idx_split()
            idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]

            data_y = torch.as_tensor(ogb_dataset.labels).squeeze(1)
            data_x = torch.as_tensor(ogb_dataset.graph['node_feat'])
            edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
            num_nodes = ogb_dataset.graph['num_nodes']
            adj = sp.coo_matrix(
                (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                shape=(num_nodes, num_nodes),
                dtype=np.float32,
            )
            if torch.is_floating_point(data_y):
                data_y = torch.nan_to_num(data_y, nan=-1.0)
            else:
                nan_mask = torch.isnan(data_y)
                if nan_mask.any():
                    data_y = data_y.masked_fill(nan_mask, -1)
            data_y = data_y.to(torch.long)

            # normalized_adj = adj_normalize(adj)
            # column_normalized_adj = column_normalize(adj)
        
        elif dataset_name in ["ogbn-arxiv", "ogbn-products"]:
            ogb_dataset = NodePropPredDataset(name=dataset_name, root=dataset_dir)
            split_idx = ogb_dataset.get_idx_split()
            idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
            
            data_y = torch.as_tensor(ogb_dataset.labels).squeeze(1)
            data_x = torch.as_tensor(ogb_dataset.graph['node_feat'])
            edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
           
            num_nodes=ogb_dataset.graph['num_nodes']
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                    shape=(num_nodes, num_nodes), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            # column_normalized_adj = column_normalize(adj)

        elif dataset_name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
            pyg_name = dataset_name.capitalize()
            pyg_data = HeterophilousGraphDataset(root=dataset_dir, name=pyg_name)[0]
            data_x = pyg_data.x.to(torch.float32)
            data_y = pyg_data.y.to(torch.long).view(-1)
            edge_index = pyg_data.edge_index.to(torch.long)
            num_nodes = int(data_x.size(0))
            adj = sp.coo_matrix(
                (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                shape=(num_nodes, num_nodes),
                dtype=np.float32,
            )
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
            if not hasattr(pyg_data, "train_mask") or not hasattr(pyg_data, "val_mask") or not hasattr(pyg_data, "test_mask"):
                raise ValueError(f"{pyg_name} dataset does not provide default train/val/test masks.")
            split_idx = {
                "train": _mask_to_index(pyg_data.train_mask, split_id=split_id).to(torch.long),
                "valid": _mask_to_index(pyg_data.val_mask, split_id=split_id).to(torch.long),
                "test": _mask_to_index(pyg_data.test_mask, split_id=split_id).to(torch.long),
            }

        # sp.save_npz(dataset_dir + dataset_name + '/adj.npz', adj)
        # sp.save_npz(dataset_dir + dataset_name + '/normalized_adj.npz', normalized_adj)
        dataset_dir = './dataset/'
        if 'split_idx' in locals():
            torch.save(
                {
                    "train": torch.as_tensor(split_idx["train"], dtype=torch.long),
                    "valid": torch.as_tensor(split_idx["valid"], dtype=torch.long),
                    "test": torch.as_tensor(split_idx["test"], dtype=torch.long),
                },
                dataset_dir + dataset_name + '/split_idx.pt'
            )
        torch.save(data_x, dataset_dir + dataset_name + '/x.pt')
        torch.save(data_y, dataset_dir + dataset_name + '/y.pt')
        torch.save(edge_index, dataset_dir + dataset_name + '/edge_index.pt')
        # sp.save_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz', column_normalized_adj)


def process_data(dataset_name, k1):
    """
    Arguments:
        k1: sequence length-1 / number of sampled neighbors
    """
    print(f'Process dataset {dataset_name}...')

    dataset_dir = './dataset/'
    data_x = torch.load(dataset_dir + dataset_name + '/x.pt')
    data_y = torch.load(dataset_dir + dataset_name + '/y.pt')
    adj = sp.load_npz(dataset_dir + dataset_name + '/adj.npz')
    normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/normalized_adj.npz')
    column_normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz')
    
    # # Compute SPD, spacial pos
    # N = adj.shape[0]
    # adj_bool = torch.zeros([N, N], dtype=torch.bool) 
    # adj_bool[data.edge_index[0, :], data.edge_index[1, :]] = True
    # shortest_path_result, path = algos.floyd_warshall(adj_bool.numpy())
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    # print(spatial_pos, spatial_pos.shape)
    # exit(0)
    
    c = 0.15
    # k1 = 100 # number of sampled neighbors, sequence length here
    Samples = 1 # sampled subgraphs for each node
    power_adj_list = [normalized_adj]
    for m in range(5): # attn_bias_dim - 1
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])

    sampling_matrix = c * inv((sp.eye(adj.shape[0]) - (1 - c) * normalized_adj).toarray()) # power_adj_list[1].toarray(), [n_node, n_node]
    # sampling_matrix = power_adj_list[4].toarray()

    # Create subgraph samples
    data_list = []
    for id in range(data_y.shape[0]):
        s = sampling_matrix[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-k1:]

        s = sampling_matrix[id]
        s[id] = 0
        s = np.maximum(s, 0)
        sample_num1 = np.minimum(k1, (s > 0).sum())
        sub_data_list = []
        for _ in range(Samples):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(data_y.shape[0]), size=sample_num1, replace=False, p=s/s.sum())
            else:
                sample_index1 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int), torch.tensor(top_neighbor_index[: k1-sample_num1], dtype=int)])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)

            sub_data_list.append([attn_bias, node_feature_id, data_y[node_feature_id].long()])
        data_list.append(sub_data_list)

    data_file_path = dataset_dir + dataset_name + '/data_s' + str(k1) + '.pt'
    torch.save(data_list, data_file_path)

    print(f'Process done!')


def rand_nodes_seq(dataset_name, k1, p=None):
    # random nodes in sequence do not overlap
    print('Generate long sequence as input')

    dataset_dir = './dataset/'
    data_x = torch.load(dataset_dir + dataset_name + '/x.pt')
    data_y = torch.load(dataset_dir + dataset_name + '/y.pt')
    adj = sp.load_npz(dataset_dir + dataset_name + '/adj.npz')
    normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/normalized_adj.npz')
    column_normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz')

    # elif args.dataset_name in ['arxiv']:
    #     dataset = DglNodePropPredDataset(name='ogbn-arxiv',
    #                                      root=dataset_dir)
    #     split_idx = dataset.get_idx_split()
    #     train, val, test = split_idx["train"], split_idx["valid"], split_idx["test"]
    #     g, labels = dataset[0]
    #     features = g.ndata['feat']
    #     nclass = 40
    #     labels = labels.squeeze()
    #     g = dgl.to_bidirected(g)

    power_adj_list = [normalized_adj]
    attn_bias_dim = 6
    for m in range(attn_bias_dim - 1): # attn_bias_dim - 1
        power_adj_list.append(power_adj_list[0]*power_adj_list[m])
    
    feature = data_x
    data_list = []

    # Shuffle node ids
    node_idx = np.arange(data_y.shape[0])
    random.shuffle(node_idx)

    # Each group contains random nodes for train
    n_group = math.ceil(data_y.shape[0]/k1)
    for group in range(n_group):
        sub_data_list = []
        for _ in range(1):
            if group == n_group - 1:
                # TODO: Pad length if use bs
                node_feature_id = torch.cat([torch.tensor(node_idx[group * k1: ], dtype=int)])

            else:
                node_feature_id = torch.cat([torch.tensor(node_idx[group * k1: (group+1) * k1], dtype=int)])

            attn_bias = torch.cat([torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)  # [n_node, n_node, attn_bias_dim]
            sub_data_list.append([attn_bias, node_feature_id, data_y[node_feature_id].long()])
            
        data_list.append(sub_data_list)


    data_file_path = dataset_dir + dataset_name + '/data_rand' + str(k1) + '.pt'
    # feature_file_path = dataset_dir + args.dataset_name  + '/feature.pt'
    if not os.path.exists(data_file_path): 
        torch.save(data_list, data_file_path)
    print(f'Process done!')
    # if not os.path.exists(feature_file_path): 
    #     torch.save(feature, feature_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split_id', type=int, default=0,
                        help='split column to export for datasets with multi-split masks')
    args = parser.parse_args()
    get_dataset(args.dataset, split_id=args.split_id)
