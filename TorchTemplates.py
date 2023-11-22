import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SimpleBinaryClassificationNN(nn.Module):
    def __init__(self,
                 layer_dict,
                 l1_lambda,
                 l2_lambda,
                 lossfun,
                 optimizer):
        super(SimpleBinaryClassificationNN, self).__init__()
        
        self.layers = nn.ModuleDict()

        self.lossfun = lossfun
        self.optimizer = optimizer
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda