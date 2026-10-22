"""
Adapter script to load the GNNMultiFix datasets & splits for CorGCN

Data loader returns:
    graph  : dgl.DGLGraph with ndata['feat'] (float32 [N,F]) and ndata['label'] (long [N,C])
    splits : (train_idx, val_idx, test_idx) as LongTensors -> uses same splits as GNNMultiFix
"""

import os.path as osp

import dgl
import numpy as np
import scipy.io
import torch

MF_DATASETS = ["pcg", "humloc", "eukloc", "yelp", "blogcatalog", "hyper", "dblp"]

# --------- HELPER FUNCTIONS --------------------
def _to_idx(x):
    x = torch.as_tensor(x)
    if x.dtype == torch.bool:
        x = x.nonzero(as_tuple=True)[0]
    return x.long()


def _load_split(file_path):
    masks = torch.load(file_path, map_location="cpu")
    return _to_idx(masks["train_mask"]), _to_idx(masks["val_mask"]), _to_idx(masks["test_mask"])


def _symmetrize_pairs(edge_list):
    """[E,2] -> [2, 2E] with both directions (same as GNNMultiFix humloc/eukloc/dblp)."""
    edge_list = torch.as_tensor(edge_list).long()
    return torch.cat([edge_list.t(), edge_list[:, [1, 0]].t()], dim=1)


def _build(features, labels, edge_index, splits, name):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels).float().round().long()
    edge_index = torch.as_tensor(edge_index).long()
    n = features.shape[0]
    assert labels.shape[0] == n, f"{name}: {labels.shape[0]} labels vs {n} nodes"
    assert int(edge_index.max()) < n, f"{name}: edge index out of range"

    # num_nodes is passed explicitly: otherwise DGL infers it from the max edge id
    # and silently drops trailing isolated nodes.
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=n)
    g.ndata["feat"] = features
    g.ndata["label"] = labels

    tr, va, te = splits
    assert len(set(tr.tolist()) & set(te.tolist())) == 0, "train/test overlap"
    # assert len(set(tr.tolist()) & set(va.tolist())) == 0, "train/val overlap"
    # assert len(set(va.tolist()) & set(te.tolist())) == 0, "val/test overlap"
    unl = int((labels[tr].sum(1) == 0).sum())
    print(f"[{name}] nodes={n} edges={edge_index.shape[1]} feat_dim={features.shape[1]} "
          f"classes={labels.shape[1]} | train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    if unl > 0:
        print(f"[{name}] WARNING: {unl} train nodes have NO label -> CorGCN's losses divide by "
              f"the label count and will produce NaN.")
    return g, splits

# --------- DATA LOADERS --------------------
def _load_pcg(root, split_name, train_percent):
    d = "pcg_removed_isolated_nodes"
    labels = np.genfromtxt(osp.join(root, d, "labels.csv"), delimiter=",")
    feats = np.genfromtxt(osp.join(root, d, "features.csv"), delimiter=",")
    edges = np.genfromtxt(osp.join(root, d, "edges_undir.csv"), delimiter=",")
    edge_index = torch.as_tensor(edges).long().t()
    splits = _load_split(osp.join(root, f"{d}_{train_percent}", split_name))
    return _build(feats, labels, edge_index, splits, "pcg")

def _read_edges_auto(edge_path):
    """Reads homophily-level edge files.
    Matches updated load_hyper_data GNNMultiFix scripts >> data_loader.py."""
    with open(edge_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "," if "," in first_line else None
    edges = np.genfromtxt(edge_path, dtype=np.float32, delimiter=delimiter, ndmin=2)
    if edges.shape[1] < 2:
        raise ValueError(f"Expected two edge columns in {edge_path}")
    return torch.as_tensor(edges[:, :2], dtype=torch.long).t().contiguous()

def _load_hyper(root, split_name, train_percent, homo_level, feature_noise_ratio):
    d = "Hyperspheres_10_10_0"
    labels = np.genfromtxt(osp.join(root, d, "labels.csv"), skip_header=1, dtype=np.dtype(float), delimiter=",")
    feats = np.genfromtxt(osp.join(root, d, "features.csv"), skip_header=1, dtype=np.dtype(float), delimiter=",")

    if feature_noise_ratio is not None and homo_level is None:
        # feature-quality experiment: drop relevant features, edges fixed to homo02
        start_col = int(10.0 * (round(1.0 - feature_noise_ratio, 1)))
        feats = feats[:, start_col:]
        edge_file = "homo02.txt"
        edges = torch.tensor(np.genfromtxt(osp.join(root, d, edge_file),
                                                   dtype=np.float32, delimiter=",")).long()
        edge_index = torch.transpose(edges, 0, 1)
    elif homo_level is not None and feature_noise_ratio is None:
        # homophily experiment: only the 10 relevant features
        feats = feats[:, :10]
        edge_file = homo_level + ".txt"
        edge_index = _read_edges_auto(osp.join(root, d, edge_file))
    else:
        raise ValueError("hyper: pass exactly one of --homo_level or --feature_noise_ratio "
                         "(same behaviour as GNNMultiFix's load_hyper_data)")

    splits = _load_split(osp.join(root, f"{d}_{train_percent}", split_name))
    return _build(feats, labels, edge_index, splits, f"hyper/{edge_file}")


def _load_go(root, d, tag):
    """humloc (HumanGo) and eukloc (EukaryoteGo): one fixed split.pt, no split_name."""
    el = np.genfromtxt(osp.join(root, d, "edge_list.csv"), skip_header=1, delimiter=",")[:, :2]
    edge_index = _symmetrize_pairs(el)
    labels = np.genfromtxt(osp.join(root, d, "labels.csv"), delimiter=",")
    feats = np.genfromtxt(osp.join(root, d, "features.csv"), delimiter=",")
    splits = _load_split(osp.join(root, d, "split.pt"))
    return _build(feats, labels, edge_index, splits, tag)


def _load_blogcatalog(root, split_name, train_percent):
    mat = scipy.io.loadmat(osp.join(root, "blogcatalog", "blogcatalog.mat"))
    labels = mat["group"].toarray()
    adj = mat["network"].tocoo()
    adj.eliminate_zeros()
    edge_index = torch.as_tensor(np.vstack([adj.row, adj.col])).long()
    # GNNMultiFix uses identity features (featureless graph); kept identical for comparability.
    feats = torch.eye(labels.shape[0])
    splits = _load_split(osp.join(root, f"blogcatalog_{train_percent}", split_name))
    return _build(feats, labels, edge_index, splits, "blogcatalog")


def _load_yelp(root, split_name, train_percent):
    from torch_geometric.datasets import Yelp
    data = Yelp(root=osp.join(root, "yelp"))[0]
    splits = _load_split(osp.join(root, f"yelp_{train_percent}", split_name))
    return _build(data.x, data.y, data.edge_index, splits, "yelp")


def _load_dblp(root, split_name, train_percent):
    d = osp.join(root, "dblp")
    feats = np.genfromtxt(osp.join(d, "features.txt"), delimiter=",")
    labels = np.genfromtxt(osp.join(d, "labels.txt"), delimiter=",")
    el = np.genfromtxt(osp.join(d, "dblp.edgelist"))
    edge_index = _symmetrize_pairs(el)
    splits = _load_split(osp.join(d, f"dblp_{train_percent}", split_name))
    return _build(feats, labels, edge_index, splits, "dblp")


def load_mf_dataset(data_name, root="data/", split_name="split_0.pt", train_percent=0.6,
                    homo_level=None, feature_noise_ratio=None):
    if data_name == "pcg":
        return _load_pcg(root, split_name, train_percent)
    if data_name == "hyper":
        return _load_hyper(root, split_name, train_percent, homo_level, feature_noise_ratio)
    if data_name == "humloc":
        return _load_go(root, "HumanGo", "humloc")
    if data_name == "eukloc":
        return _load_go(root, "EukaryoteGo", "eukloc")
    if data_name == "blogcatalog":
        return _load_blogcatalog(root, split_name, train_percent)
    if data_name == "yelp":
        return _load_yelp(root, split_name, train_percent)
    if data_name == "dblp":
        return _load_dblp(root, split_name, train_percent)
    raise ValueError(f"unknown GNNMultiFix dataset: {data_name}")
