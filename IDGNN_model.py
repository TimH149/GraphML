import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class IGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(IGNNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix
        x = self.lin(x)

        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index, size):
        # Compute normalization
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

class IGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(IGNN, self).__init__()
        self.conv1 = IGNNConv(in_dim, hidden_dim)
        self.conv2 = IGNNConv(hiddem_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)