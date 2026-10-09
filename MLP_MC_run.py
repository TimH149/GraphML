import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch.nn import Linear

class MLP(torch.nn.Module):
    def __init__(self, dataset):
        super(MLP, self).__init__()
        self.lin1 = Linear(dataset.num_features, 32)
        self.lin2 = Linear(32, dataset.num_classes)

    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

# Load the Cora/Citeseer dataset
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(dataset).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))