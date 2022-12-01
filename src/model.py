import torch

from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.dense import dense_diff_pool

from utils import construct_mask, my_dense_diff_pool

class DiffPool(torch.nn.Module):
    '''
    DiffPool:Hierarchical Graph Representation Learning with Differentiable Pooling
    '''
    def __init__(self, args, number_of_labels, num_y) -> None:
        super().__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.num_y = num_y
        self.device = torch.device(self.args.gpu_id)

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
        bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def build_conv_layers(self, input_dim, hidden_dim, output_dim):
        '''
        Construct a three layer.
        '''
        conv_1 = DenseGCNConv(in_channels = input_dim,  out_channels = hidden_dim)
        conv_2 = DenseGCNConv(in_channels = hidden_dim, out_channels = hidden_dim)
        conv_3 = DenseGCNConv(in_channels = hidden_dim, out_channels = output_dim)
        return conv_1, conv_2, conv_3
    
    def setup_layers(self):
        # Input GCN
        self.GCN_input  = DenseGCNConv(self.number_of_labels, self.args.hidden_dim)
        self.GCN_hidden = DenseGCNConv(self.args.hidden_dim, self.args.hidden_dim)
        self.GCN_output = DenseGCNConv(self.args.hidden_dim, self.args.hidden_dim)

        # Assign GCN
        self.GCN_ass_input  = torch.nn.ModuleList()
        self.GCN_ass_hidden = torch.nn.ModuleList()
        self.GCN_ass_output = torch.nn.ModuleList()
        self.Assign_fc      = torch.nn.ModuleList()
        assign_dim = int(self.args.max_num_node * self.args.assign_ratio)
        assign_dims = []
        assign_input_dim = self.number_of_labels

        for id in range(self.args.num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_1, assign_conv_2, assign_conv_3 = self.build_conv_layers(assign_input_dim, self.args.hidden_dim, assign_dim)
            assign_linear = torch.nn.Linear( self.args.hidden_dim*2 + assign_dim, assign_dim)

            self.GCN_ass_input.append(assign_conv_1)
            self.GCN_ass_hidden.append(assign_conv_2)
            self.GCN_ass_output.append(assign_conv_3)
            self.Assign_fc.append(assign_linear)

            # next layer set
            assign_dim = int(assign_dim * self.args.assign_ratio)
            assign_input_dim = self.args.hidden_dim*3

        # Output GCN
        self.GCN_out_input  = torch.nn.ModuleList()
        self.GCN_out_hidden = torch.nn.ModuleList()
        self.GCN_out_output = torch.nn.ModuleList()

        for id in range(self.args.num_pooling):
            conv_1, conv_2, conv_3 = self.build_conv_layers(3*self.args.hidden_dim, self.args.hidden_dim, self.args.hidden_dim)
            self.GCN_out_input.append(conv_1)
            self.GCN_out_hidden.append(conv_2)
            self.GCN_out_output.append(conv_3)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear((self.args.hidden_dim*3) * (self.args.num_pooling+1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.num_y)
        )
    
    def gcn_forward(self, x, adj, input_layer, hidden_layer, output_layer):
        '''
        return: [batch, node_num, 2 * hidden_dim + output_dim]
        '''
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
    
    def forward(self, feat, adj, node_num = None):
        loss = 0
        ass_feat = feat
        out_all = []

        feat = self.gcn_forward(feat, adj, self.GCN_input, self.GCN_hidden, self.GCN_output)

        out, _ = torch.max(feat, dim=1)
        out_all.append(out)

        for i in range(self.args.num_pooling):
            if node_num != None and i == 0:
                embedding_mask = construct_mask(self.args.max_num_node, node_num).to(self.device)
            else:
                embedding_mask = None


            ass_feat = self.gcn_forward(ass_feat, adj, self.GCN_ass_input[i], self.GCN_ass_hidden[i], self.GCN_ass_output[i])
            ass_feat = self.Assign_fc[i](ass_feat)

            if embedding_mask != None:
                ass_feat = ass_feat * embedding_mask

            feat, adj, loss_lp, loss_e = my_dense_diff_pool(feat, adj, ass_feat)
            loss += loss_lp + loss_e

            feat = self.gcn_forward(feat, adj, self.GCN_out_input[0], self.GCN_out_hidden[0], self.GCN_out_output[0])
            ass_feat = feat

            out, _ = torch.max(feat, dim=1)
            out_all.append(out)

        feat = torch.cat(out_all, dim=1)
        score = self.fc(feat)

        return score, loss
