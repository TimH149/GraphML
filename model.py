import torch.nn as nn
from torch_geometric.nn import GCNConv
from metrics import *
from torch.nn import functional as F

class FPLPGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10):
        super(FPLPGCN, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Linear(hidden_dim + output_dim, output_dim)

        self.smd = torch.nn.Sigmoid()

    def forward(self, x, y, edge_index):
        # GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)

        # Fusion of feature propagation and label propagation
        x_fused = torch.cat((x, x_label), dim=1)
        x_fused = self.fusion_layer(x_fused)

        return self.smd(x_fused)



class Local_Glbal_LC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10):
        super(Local_Glbal_LC, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Linear(hidden_dim + output_dim, output_dim)

        self.sgmd = torch.nn.Sigmoid()

    def forward(self, x, y, edge_index, LC_matrix):

        # GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        # local label correlation
        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)
                print("activated")

        # Fusion of feature propagation and label propagation
        x_fused = torch.cat((x, x_label), dim=1)
        x_fused = self.fusion_layer(x_fused)
        # global label correlation: label co-occurence
        x_fused = torch.mm(x_fused, LC_matrix)

        return self.sgmd(x_fused)




class FPLPGCN_dw(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64, multi_class=False):
        super(FPLPGCN_dw, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Linear(hidden_dim + output_dim + dw_dim, output_dim)
        self.fuse_FP_dw = nn.Linear(hidden_dim + dw_dim, output_dim)

        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

    def forward(self, x, y, edge_index, deep_walk_emb):

        # GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)


        x_fused = torch.cat((x, x_label, deep_walk_emb), dim=1)
        self.all_emb = x_fused
        x_fused = self.fusion_layer(x_fused)

        if not self.multi_class:
            return self.smd(x_fused)
        else:
            return x_fused


class FPLPGCN_dw_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64, multi_class=False):
        super(FPLPGCN_dw_MLP, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Sequential(
                                          nn.Linear(hidden_dim + output_dim + dw_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, output_dim)
                                         )

        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

    def forward(self, x, y, edge_index, deep_walk_emb):

        # GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)


        x_fused = torch.cat((x, x_label, deep_walk_emb), dim=1)
        self.all_emb = x_fused
        x_fused = self.fusion_layer(x_fused)

        if not self.multi_class:
            return self.smd(x_fused)
        else:
            return x_fused
    



class FPLPGCN_dw_linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64, multi_class=False):
        super(FPLPGCN_dw_linear, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Linear(hidden_dim + output_dim + dw_dim, output_dim)
        ################# for test only the label propagation #################
        #self.fusion_layer = nn.Linear(output_dim+dw_dim, output_dim)
        #####################################################################################
        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

        self.FP = None
        self.LP = None
        self.dw_emb = None
        self.all_emb = None

    def forward(self, x, y, edge_index, deep_walk_emb):

        #GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
        self.FP = x
        # print("FP: ", self.FP)

        ####################################################################
        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
        self.LP = x_label
        # print("LP: ", self.LP)
        # adj = SparseTensor(row=edge_index[0], col=edge_index[1])
        # dense_adj = adj.to_dense()
        # # row normalize adj
        # row_sums = dense_adj.sum(dim=1)
        # normalized_adj = dense_adj / row_sums.unsqueeze(1)

        # gcn normalize adj

        # # Add self-loops to the adjacency matrix
        # adj = adj.set_diag()
        # deg = adj.sum(dim=1).pow(-0.5)

        # # Normalize the adjacency matrix
        # normalized_adj = adj * deg.view(-1, 1) * deg.view(1, -1)
        # normalized_adj = normalized_adj.to_dense()

        # x_label = torch.mm(normalized_adj, x_label)
        # print("input label matrix",x_label)
        # x_label = torch.mm(dense_adj, x_label)

        #print("before norm:", x_label)
        # row_sums = x_label.sum(dim=1)
        # x_label = x_label / row_sums.unsqueeze(1)
        # print("after norm x_label: ", x_label)
        ####################################################################


        self.dw_emb = deep_walk_emb
        # print("dw emb", self.dw_emb)
        ####################################################################
        # Fusion of feature propagation and label propagation
        x_fused = torch.cat((x, x_label, deep_walk_emb), dim=1)
        
        #x_fused = torch.cat((x_label, deep_walk_emb), dim=1)
        #x_fused = x_label

        ####################################################################
        self.all_emb = x_fused
        x_fused = self.fusion_layer(x_fused)

        if not self.multi_class:
            ####################################################################
            return self.smd(x_fused)
            # # calculate ap
            # print(x_fused)
            # return x_fused
            ####################################################################
        else:
            return x_fused


class FPLPGCN_dw_weighted(nn.Module):
    """GNN-MultiFix with learned per-node weighting of the feature, label and positional modules.

    Each module representation h^m is projected to a common space z^m = P_m h^m, scored with a
    shared scorer s^m = q^T tanh(W_a z^m + b_a), and the scores are turned into weights
    w = softmax(s^f, s^l, s^p). The classifier then sees [w^f h^f; w^l h^l; w^p h^p].

    With gate_norm=True each module is instead mapped to a shared space and normalised,
    u^m = LayerNorm(F_m h^m) without a learnable scale, and the classifier sees the weighted sum
    w^f u^f + w^l u^l + w^p u^p. The classifier can then no longer rescale one module to undo
    its weight, so the weights are the only per-module scale and can be read as module importance.

    With nonlinear=True a ReLU follows every GCN layer except the last one of each module, as in
    FPLPGCN_dw (MLP-1); otherwise the modules are linear, as in FPLPGCN_dw_linear.

    weight_scale multiplies the softmax weights before they are applied. With weight_scale=3, equal
    weights are 1 instead of 1/3, so the modules enter the output layer at the same scale as in plain
    MultiFix. module_weights always stores the softmax weights themselves (they sum to 1).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64,
                 gate_dim=16, gate_norm=False, nonlinear=False, weight_scale=1.0, multi_class=False):
        super(FPLPGCN_dw_weighted, self).__init__()
        self.nonlinear = nonlinear
        self.weight_scale = weight_scale

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # projections P_f, P_l, P_p to the common gate space
        self.proj_fp = nn.Linear(hidden_dim, gate_dim, bias=False)
        self.proj_lp = nn.Linear(output_dim, gate_dim, bias=False)
        self.proj_dw = nn.Linear(dw_dim, gate_dim, bias=False)

        # shared scorer: W_a, b_a and q
        self.score_hidden = nn.Linear(gate_dim, gate_dim)
        self.score_vector = nn.Linear(gate_dim, 1, bias=False)

        self.gate_norm = gate_norm
        if gate_norm:
            # F_f, F_l, F_p to the shared fusion space, normalised without a learnable scale
            self.fuse_fp = nn.Linear(hidden_dim, hidden_dim)
            self.fuse_lp = nn.Linear(output_dim, hidden_dim)
            self.fuse_dw = nn.Linear(dw_dim, hidden_dim)
            self.fuse_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.fusion_layer = nn.Linear(hidden_dim, output_dim)
        else:
            # Linear layer for fusion
            self.fusion_layer = nn.Linear(hidden_dim + output_dim + dw_dim, output_dim)
        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

        # (num_nodes, 3) weights for [feature, label, positional] from the last forward pass
        self.module_weights = None

    def score(self, z):
        return self.score_vector(torch.tanh(self.score_hidden(z)))

    def forward(self, x, y, edge_index, deep_walk_emb):

        #GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if self.nonlinear and i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if self.nonlinear and j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)

        # one score per module, softmax over the three modules for every node
        scores = torch.cat((self.score(self.proj_fp(x)),
                            self.score(self.proj_lp(x_label)),
                            self.score(self.proj_dw(deep_walk_emb))), dim=1)
        weights = F.softmax(scores, dim=1)
        self.module_weights = weights.detach()
        weights = self.weight_scale * weights

        if self.gate_norm:
            x_fused = (weights[:, 0:1] * self.fuse_norm(self.fuse_fp(x))
                       + weights[:, 1:2] * self.fuse_norm(self.fuse_lp(x_label))
                       + weights[:, 2:3] * self.fuse_norm(self.fuse_dw(deep_walk_emb)))
        else:
            x_fused = torch.cat((weights[:, 0:1] * x,
                                 weights[:, 1:2] * x_label,
                                 weights[:, 2:3] * deep_walk_emb), dim=1)
        self.all_emb = x_fused
        x_fused = self.fusion_layer(x_fused)

        if not self.multi_class:
            return self.smd(x_fused)
        else:
            return x_fused


