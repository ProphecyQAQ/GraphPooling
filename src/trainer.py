import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset

from model import DiffPool
from datasets import GraphDataset
from paramParse import paramterParser

from tqdm import tqdm, trange

class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(self.args.gpu_id)
        self.train_data, self.test_data = self.prepareData()

        self.model = DiffPool(args, self.num_x, self.num_y).to(self.device)

    def prepareData(self):
        data = TUDataset("./datasets/{}.".format(self.args.dataset), name = self.args.dataset, use_node_attr=True)
        self.num_x = data[0].x.shape[1]

        y_num = set()
        max_num_nodes = 0
        for g in data:
            max_num_nodes = max(max_num_nodes, g.x.shape[0])
            y_num.add(int(g.y[0]))
        self.num_y = len(y_num)
        print("Max node num is {}".format(max_num_nodes))

        data = data.shuffle()
        train_data = data[:int(len(data)*0.8)]
        test_data  = data[int(len(data)*0.8):] 

        train_dataset = GraphDataset(train_data, max_num_nodes)
        train_loader  = DataLoader(train_dataset, batch_size = self.args.batch_size)

        test_dataset  = GraphDataset(test_data, max_num_nodes)
        test_loader   = DataLoader(test_dataset, batch_size = self.args.batch_size)
        return train_loader, test_loader
        
    def fit(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.args.lr, 
            weight_decay = self.args.weight_decay)

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

                y_pred, loss_ = self.model(x, adj)
                loss = F.cross_entropy(y_pred.cpu(), y.view(-1), reduction='mean') + loss_
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                optimizer.step()
                loss_sum += loss
                num += len(y.view(-1))

            loss = loss_sum/num
            epochs.set_description("Epoch (Loss=%g)" % loss)

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        for idx, graph in tqdm(enumerate(self.test_data), total=len(self.test_data),  desc="Batches", leave=False):
            x = graph['x'].to(self.device)
            adj = graph['adj'].to(self.device)
            y = graph['y']

            y_pred, _ = self.model(x, adj)
            prediction = torch.argmax(y_pred, 1).cpu()
            correct += (prediction == y.view(-1)).sum()
            total += len(y.view(-1)) 
        print((correct/total).detach().numpy())