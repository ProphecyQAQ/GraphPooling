import torch

from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.dense import dense_diff_pool

class DiffPool(torch.nn.Module):
    '''
    DiffPool:Hierarchical Graph Representation Learning with Differentiable Pooling
    '''
    def __init__(self, args, number_of_labels, num_y) -> None:
        super().__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.num_y = num_y

        self.act = torch.nn.ReLU()
        self.setup_layers()
        self.init_weight()

    def init_weight(self):
        '''
        Init model paramaters
        '''
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)
    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = torch.nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)
    
    def setup_layers(self):
        # Input GCN
        self.GCN_input  = DenseGCNConv(self.number_of_labels, 64)
        self.GCN_hidden = DenseGCNConv(64, 64)
        self.GCN_output = DenseGCNConv(64, 64)

        # Assign GCN
        self.GCN_ass_input  = DenseGCNConv(self.number_of_labels, 64)
        self.GCN_ass_hidden = DenseGCNConv(64, 64)
        self.GCN_ass_output = DenseGCNConv(64, 100)

        # Output GCN
        self.GCN_out_input  = DenseGCNConv(192, 64)
        self.GCN_out_hidden = DenseGCNConv(64, 64)
        self.GCN_out_output = DenseGCNConv(64, 64)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(192, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.num_y)
        )
    
    def gcn_forward(self, x, adj, input_layer, hidden_layer, output_layer):
        x_all = []

        x = input_layer(x, adj)
        x = self.act(x)
        if self.args.batch_norm:
            x = self.apply_bn(x)
        x_all.append(x)

        x = hidden_layer(x, adj)
        x = self.act(x)
        if self.args.batch_norm:
            x = self.apply_bn(x)
        x_all.append(x)

        x = output_layer(x, adj)
        x_all.append(x)

        x_tensor = torch.cat(x_all, dim = 2)

        return x_tensor
    
    def forward(self, feat, adj):
        loss = 0
        ass_feat = feat

        feat = self.gcn_forward(feat, adj, self.GCN_input, self.GCN_hidden, self.GCN_output)
        ass_feat = self.gcn_forward(ass_feat, adj, self.GCN_ass_input, self.GCN_ass_hidden, self.GCN_ass_output)

        feat, adj, loss_lp, loss_e = dense_diff_pool(feat, adj, ass_feat)
        loss += loss_lp + loss_e

        feat = self.gcn_forward(feat, adj, self.GCN_out_input, self.GCN_out_hidden, self.GCN_out_output)

        score, _ = torch.max(feat, dim=1)
        score = self.fc(score)

        return score, loss
