# %%
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
from src.model import DiffPool

import random
import numpy as np

# %%
data = TUDataset("datasets/PROTEINS.", name = "PROTEINS")

# %%
max_num_nodes = 0
for g in data:
    max_num_nodes = max(max_num_nodes, g.x.shape[0])

# %%
f = torch.zeros((max_num_nodes, 3))
f[:g.x.shape[0], :g.x.shape[1]] = g.x

# %%
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_num_nodes = None) -> None:
        super().__init__()
        self.adj_list        = []
        self.x_list          = []
        self.y_list          = []
        self.edge_index_list = []
        self.max_num_nodes = max_num_nodes
        self.prepareData(data, max_num_nodes)
    
    def prepareData(self, data, max_num_nodes = None):
        for g in data:
            f = torch.zeros((self.max_num_nodes, g.x.shape[1]))
            f[:g.x.shape[0], :g.x.shape[1]] = g.x
            self.x_list.append(f)
            self.y_list.append(g.y)
            self.edge_index_list.append(g.edge_index)
            adj = to_dense_adj(g.edge_index)
            self.adj_list.append(adj[0])

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, idx):
        adj = self.adj_list[idx]
        num_nodes = adj.shape[0]
        adj_padded = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        #adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        #adj_padded[:num_nodes, :num_nodes] = adj

        return {'adj':adj_padded,
                'x':self.x_list[idx],
                'y':self.y_list[idx],
                'num_nodes': num_nodes 
                }


# %%
data = data.shuffle()

# %%
train_data = data[:int(len(data)*0.8)]
test_data  = data[int(len(data)*0.8):] 

# %%
train_dataset = GraphDataset(train_data, max_num_nodes)
train_loader  = DataLoader(train_dataset, batch_size = 2)

test_dataset  = GraphDataset(test_data, max_num_nodes)
test_loader   = DataLoader(test_dataset, batch_size = 2)

# %%
device = torch.device("cuda:0")
model = DiffPool(None, number_of_labels= 3).to(device)
optimizer = torch.optim.Adam(model.parameters() ,lr = 1e-4)

# %%
model.train()
for epoch in range(100):
    loss_sum = 0
    total = 0
    for idx, graph in enumerate(train_loader):
        optimizer.zero_grad()

        x = graph['x'].to(device)
        adj = graph['adj'].to(device)
        y = graph['y']

        y_pred = model(x, adj)
        loss = F.cross_entropy(y_pred.cpu(), y.view(-1), reduction='mean')
        loss.backward()
        optimizer.step()
        loss_sum += loss
        total += len(y.view(-1))
    print(loss_sum/total)
    loss_sum = 0
    total = 0

# %%
model.eval()
correct = 0
total = 0
for idx, graph in enumerate(test_loader):
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    y = graph['y']

    y_pred = model(x, adj)
    prediction = torch.argmax(y_pred, 1).cpu()
    correct += (prediction == y.view(-1)).sum()
    total += len(y.view(-1)) 
print((correct/total).detach().data.numpy())


