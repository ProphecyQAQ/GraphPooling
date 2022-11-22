import torch
from torch_geometric.datasets import TUDataset

class DataSets():
    '''
    Datasets pre-process class
    '''

    def __init__(self, args) -> None:
        self.args = args
        
