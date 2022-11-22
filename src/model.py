import torch

from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.dense import dense_diff_pool

class DiffPool(torch.nn.Module):
    '''
    DiffPool:Hierarchical Graph Representation Learning with Differentiable Pooling
    '''
    def __init__(self, args, number_of_labels) -> None:
        super().__init__()
        self.args = args
        self.number_of_labels = number_of_labels

        self.act = torch.nn.ReLU()
        self.setup_layers()
    
    def setup_layers(self):
        self.GCN1 = DenseGCNConv(self.number_of_labels, 64)
        self.GCN2 = DenseGCNConv(64, 32)
        self.GCN3 = DenseGCNConv(32, 16)

        self.assign_gcn1 = DenseGCNConv(self.number_of_labels, 1000)
        self.assign_gcn2 = DenseGCNConv(1000, 100)
        self.assign_gcn3 = DenseGCNConv(100, 1)

        self.fc = torch.nn.Linear(16, 2)

    def forward(self, feat, adj):
        assign_feat = feat
        feat = self.GCN1(feat, adj)
        feat = self.act(feat)
        feat = self.GCN2(feat, adj)
        feat = self.act(feat)
        feat = self.GCN3(feat, adj)

        assign_feat = self.assign_gcn1(assign_feat, adj)
        assign_feat = self.act(assign_feat)
        assign_feat = self.assign_gcn2(assign_feat, adj)
        assign_feat = self.act(assign_feat)
        assign_feat = self.assign_gcn3(assign_feat, adj)
        feat, adj, loss_lp, loss_e = dense_diff_pool(feat, adj ,assign_feat)

        score = self.fc(feat.mean(dim=-2))
        #score = torch.sigmoid(score)
        score = torch.softmax(score, dim=-1)
        return score, loss_lp, loss_lp
