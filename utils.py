import torch
import numpy as np
from torch_geometric.utils import remove_isolated_nodes
from scipy.sparse import diags
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import scatter

def row_normalize(tensor):
    # Calculate the sum of each row
    row_sums = tensor.sum(dim=1, keepdim=True)

    # Divide each element in the tensor by its row sum
    normalized_tensor = tensor / row_sums

    return normalized_tensor


def row_normlize_sparsetensor(a):
    deg = a.to_dense().sum(dim=1).to(torch.float)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    a_n = deg_inv.view(-1, 1) * a
    return a_n


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def remove_and_reindex_isolated_nodes(G):
    # Get a mask of non-isolated nodes
    non_isolated_nodes_mask, _ = remove_isolated_nodes(G.edge_index)

    # Select non-isolated nodes
    G.x = G.x[non_isolated_nodes_mask]
    G.y = G.y[non_isolated_nodes_mask] if G.y is not None else None

    # Reindex edge indices
    node_idx_mapping = torch.where(non_isolated_nodes_mask)[0]
    edge_idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_idx_mapping.tolist())}
    G.edge_index = torch.tensor([[edge_idx_mapping[edge] for edge in edge_list] for edge_list in G.edge_index.tolist()])

    return G

def check_isolated_nodes(data):
    edge_index = data.edge_index
    all_nodes = torch.arange(data.y.shape[0])
    connected_nodes = torch.unique(edge_index)
    isolated_nodes = torch.tensor([node for node in all_nodes if node not in connected_nodes])
    return isolated_nodes

def glorot(shape):
    init_range = np.sqrt(6.0 / np.sum(shape))
    #(r1 - r2) * torch.rand(a, b) + r2
    initial = (init_range-init_range) * torch.rand(shape) + (-init_range)

    return initial


def normalize_tensor(a):
    """Row-normalize tensor that requires grad"""
    rowsum = a.sum(1).detach().cpu().numpy()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(a.detach().cpu().numpy())
    mx = torch.tensor(mx)
    return mx

def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

####################### LSPE ##########################################
def init_positional_encoding(g, pos_enc_dim, type_init="rand_walk"):
    """
        Initializing positional encoding with RWPE
        source: https://github.com/vijaydwivedi75/gnn-lspe/blob/main/data/ogb_mol.py

    """
    import dgl
    # convert to dgl graph to use the functions 
    feat = g.x
    edge_index = g.edge_index
    g = dgl.graph((g.edge_index[0], g.edge_index[1]))
    g.x = feat

    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        coo = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n))
        A = coo.tocsr()
        Dinv = diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1

        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE
    
    # return g
    return g.ndata['pos_enc']


#### IDGNN

def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                 dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1.0 if not improved else 2.0
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                       fill_value, num_nodes)

    row, col = edge_index
    deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



def compute_identity(edge_index, n, k):
    id, value = norm(edge_index, n)
    adj_sparse = torch.sparse.FloatTensor(id, value, torch.Size([n, n]))
    adj = adj_sparse.to_dense()
    diag_all = [torch.diag(adj)]
    adj_power = adj
    for i in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all


# for big graph, where it is not possible to use adj_dense


def sparse_diag(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    diag_indices = indices[0] == indices[1]
    diag_values = values[diag_indices]
    return diag_values


def compute_identity_sparse(edge_index, n, k):
    id, value = norm(edge_index, n)
    adj_sparse = torch.sparse.FloatTensor(id, value, torch.Size([n, n]))
    diag_all = [sparse_diag(adj_sparse)]
    adj_power = adj_sparse
    for i in range(1, k):
        adj_power = torch.sparse.mm(adj_power, adj_sparse)
        diag_all.append(torch.sparse.sum(adj_power, dim=0).to_dense())
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all
