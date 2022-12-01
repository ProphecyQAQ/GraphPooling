import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree

from model import DiffPool
from datasets import GraphDataset
from paramParse import paramterParser

from tqdm import tqdm, trange

class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(self.args.gpu_id)
        self.train_data, self.val_data, self.test_data = self.prepareData()

        self.model = DiffPool(args, self.num_x, self.num_y).to(self.device)

    def prepareData(self):
        data = TUDataset("./datasets/{}.".format(self.args.dataset), name = self.args.dataset, use_node_attr=True)

        if data[0].x == None:  # node don't have attribute, according to degree encode onehot for each node
            max_degree = 0

            for g in data:
                max_degree = max(
                    max_degree,
                    int(max(degree(g.edge_index[0])))
                )

            one_hot_degree = OneHotDegree(max_degree, cat=False)
            # Add ont-hot feature for each node
            data.transform = one_hot_degree


        self.num_x = data[0].x.shape[1]

        y_num = set()
        max_num_nodes = 0
        for g in data:
            max_num_nodes = max(max_num_nodes, g.x.shape[0])
            y_num.add(int(g.y[0]))
        self.num_y = len(y_num)
        max_num_nodes = self.args.max_num_node
        print("Max node num is {}".format(max_num_nodes))

        data = data.shuffle()
        train_num  = int(len(data) * 0.8)
        test_num   = int(len(data) * (1-0.1))
        train_data = data[:train_num]
        val_data   = data[train_num:test_num]
        test_data  = data[test_num:] 

        train_dataset = GraphDataset(train_data, max_num_nodes)
        train_loader  = DataLoader(train_dataset, batch_size = self.args.batch_size)

        val_dataset   = GraphDataset(val_data, max_num_nodes)
        val_loader    = DataLoader(val_dataset, batch_size = self.args.batch_size)

        test_dataset  = GraphDataset(test_data, max_num_nodes)
        test_loader   = DataLoader(test_dataset, batch_size = self.args.batch_size)
        return train_loader, val_loader, test_loader
        
    def fit(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.args.lr, 
            weight_decay = self.args.weight_decay)

        self.bets_acc = 0
        self.model.train()
        epochs = trange(self.args.epochs, leave = True, desc = "Epoch")
        for epoch in epochs:
            loss_sum = 0
            num = 0

            for idx, graph in tqdm(enumerate(self.train_data), total=len(self.train_data),  desc="Batches", leave=False):
                optimizer.zero_grad()

                x = graph['x'].to(self.device)
                adj = graph['adj'].to(self.device)
                y = graph['y']
                num_nodes = graph['num_nodes'].to(self.device)

                y_pred, loss_ = self.model(x, adj, num_nodes)
                loss = F.cross_entropy(y_pred.cpu(), y.view(-1), reduction='mean') + loss_
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                optimizer.step()
                loss_sum += loss
                num += len(y.view(-1))

            loss = loss_sum/num
            epochs.set_description("Epoch (Loss=%g)" % loss)

            if epoch%10 == 0:
                val_acc = self.eval(validate = True)
                if val_acc > self.bets_acc:
                    self.best_model = self.model
                    print("Now best val acc {}".format(val_acc))


    def eval(self, validate = False):
        correct = 0
        total = 0

        if validate == True:
            data  = self.val_data
            model = self.model
        else:
            data  = self.test_data
            model = self.best_model
        
        model.eval()

        for idx, graph in tqdm(enumerate(data), total=len(data),  desc="Batches", leave=False):
            x = graph['x'].to(self.device)
            adj = graph['adj'].to(self.device)
            y = graph['y']

            y_pred, _ = model(x, adj)
            prediction = torch.argmax(y_pred, 1).cpu()
            correct += (prediction == y.view(-1)).sum()
            total += len(y.view(-1)) 
        return (correct/total).detach().numpy()