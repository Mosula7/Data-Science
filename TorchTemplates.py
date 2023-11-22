import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SimpleBinaryClassificationNN(nn.Module):
    def __init__(self,
                 layer_dict,        # layers in the following form: ([n inputs, n outputs], batch normalization: True/False, fraction of units dropped, activation_function: False if No activation needed)
                 l1_lambda,         # L1 regularization term
                 l2_lambda,         # L1 regularization term
                 ):
        super(SimpleBinaryClassificationNN, self).__init__()
        
        self.layers = nn.ModuleDict()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.layers = nn.ModuleDict()
        for key, values in layer_dict.items():
            self.layers[f'fc_{key}'] = nn.Linear(values[0][0], values[0][1])

            if values[1]:
                self.layers[f'bn_{key}'] = nn.BatchNorm1d(values[0][1])
 
            if values[2]:
                self.layers[f'do_{key}'] = nn.Dropout(values[2])

            if values[3]:
                self.layers[f'act_{key}'] = getattr(torch.nn, values[3])()

        n_weights = 0
        for name,param in self.named_parameters():
            if 'bias' not in name:
                n_weights += param.numel()

        self.n_weights = n_weights
        self.layer_names = [name for name in self.layers]

    def forward(self, x):
        for layer in self.layer_names:
            x = self.layers[layer](x)
            return x 


    def l1_regularization(self): 
        # summing absolute values of all wights (except biases) to be used during gradient decent
        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.sum(torch.abs(param))
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l1_regularization * self.l1_lambda / self.n_weights
    

    def l2_regularization(self):
        # summing squared values of all wights (except biases) to be used during gradient decent 
        l2_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l2_regularization = l2_regularization + torch.sum(param ** 2) 
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l2_regularization * self.l2_lambda / self.n_weights

