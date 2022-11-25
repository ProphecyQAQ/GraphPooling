import torch
from torch.utils import data

from torch_geometric.utils import to_dense_adj

class GraphDataset(data.Dataset):
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
