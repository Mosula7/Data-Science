import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchmetrics
import operator


def make_loader(X: pd.DataFrame, 
                y: pd.Series, 
                batch_size: int = None, 
                shuffle: bool = True, 
                imputer: 'sklearn imputer' = None,
                scaler: 'sklearn scaler' = None) -> torch.utils.data.DataLoader:
        
    if type(X) != pd.DataFrame:
        return 'Expected X a pd.DataFrame'
    
    if not batch_size:
        batch_size = X.shape[0]
    
    if imputer:
        imputer_func = 'transform' if hasattr(imputer, "n_features_in_") else 'fit_transform'
        X = imputer.__getattribute__(imputer_func)(X)
    
    if scaler:
        scaler_func = 'transform' if hasattr(scaler, "n_features_in_") else 'fit_transform'
        X = scaler.__getattribute__(scaler_func)(X)

    data = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y.values).reshape(-1,1).float())
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader


class EarlyStopping:
    def __init__(self, early_stop, objective, verbose=True):
        if objective not in ('min', 'max'):
            raise ValueError("Expected 'min' or 'max'")
        elif objective == 'min':
            self.comp_fn = operator.lt
        elif objective == 'max':
            self.comp_fn = operator.gt

        self.verbose = verbose
        self.early_stop = early_stop
        self.no_imp = 0
        
    def __call__(self, history):
        if history and self.comp_fn(max(history), history[-1]):
            self.no_imp += 1

            if self.no_imp == self.early_stop:
                if self.verbose:
                    print(f'no improvement for {self.no_imp} rounds, early stop')
                return True
            else:
                return False

        else:
            self.no_imp = 0
            return False


class DenseLayer(nn.Module):
    def __init__(self, 
            n_inputs: int, 
            n_output: int, 
            batchnorm: bool = None, 
            dropout: float = None, 
            activation: str = None, 
            L1: float = None, 
            L2: float = None
            ) -> nn.Module:
        super().__init__()

        # ceating a fully conected linear layer
        self.linear = nn.Linear(n_inputs, n_output)

        # adding batchnormalization dropout and activation layer if specified
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(n_output)
        if activation:
            self.activation = getattr(nn, activation)()
        if dropout:
            self.dropout = nn.Dropout(dropout)

        # counting weights which L1 and L2 regularization is applied to 
        self.n_weights = 0
        for name,param in self.named_parameters():
            if ('weight' in name) and ('batchnorm' not in name):
                self.n_weights += param.numel()

        # saving lambdas
        self.l1_lambda = L1
        self.l2_lambda = L2


    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X


    def l1_regulizer(self):
        # summing absolute values of all wights (except biases) to be used during gradient decent
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if ('weight' in name) and ('bn' not in name):
                l1_reg = l1_reg + torch.sum(torch.abs(param))
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l1_reg * self.l1_lambda / self.n_weights
    

    def l2_regulizer(self):
        # summing squared values of all wights (except biases) to be used during gradient decent 
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if ('weight' in name) and ('bn' not in name):
                l2_reg = l2_reg + torch.sum(torch.square(param)) 
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l2_reg * self.l2_lambda / self.n_weights


class TabularNN(nn.Module):
    def __init__(self,
                 layers: dict
                 ) -> nn.Module:
        super(TabularNN, self).__init__()
        
        # adding layers to a module dict so they are named and organized
        self.layers = nn.ModuleDict()
        for name, layer in layers.items():
            if not isinstance(layer, DenseLayer):
                raise ValueError("Expected instance of 'DenseLayer'")
            self.layers[name] = layer


    def forward(self, x):
        for layer in self.layers:
            x = self.layers[layer](x)
        return x 

    
    def l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, layer in self.layers.items():
            if layer.l1_lambda:
                l1_reg = l1_reg + layer.l1_regulizer()
        return l1_reg
        
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, layer in self.layers.items():
            if layer.l2_lambda:
                l2_reg = l2_reg + layer.l2_regulizer()
        return l2_reg
 

    def fit(self, 
            train_loader, test_loader, 
            epochs:int,
            lossfun, optimizer, 
            eval_func, early_stopping, verbose=True):
        
        eval_metric = eval_func.__repr__().replace('()', '')

        self.eval_train = {}
        self.eval_test = []
        self.losses = {}

        # looping over all the epochs
        for epoch in range(epochs):
            self.train() # switching model back to training mode after evaluating it on the validation set
            self.eval_train[epoch] = []
            self.losses[epoch] = []
            # looping over all the batches
            for i, (X,y) in enumerate(train_loader):

                yHat = self(X) # forward pass
                loss = lossfun(yHat,y) + self.l1_regularization() + self.l2_regularization() # calculating loss
               
                # backpropagation
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()
               
                # evaluating on the current batch
                self.eval()
                with torch.no_grad():
                    yHat = self(X)
                self.train()

                # logging batch performance
                self.eval_train[epoch].append(float(eval_func(torch.sigmoid(yHat).detach(), y)))
                self.losses[epoch].append(loss.detach().item())

                if verbose:
                    print(f'batch {i} {eval_metric}: {self.eval_train[epoch][-1]:.3f} | loss: {self.losses[epoch][-1]:.3f}', end = '\r')

        
            # evaluating on the test set
            X,y = next(iter(test_loader))
            self.eval()
            with torch.no_grad():
                yHat = self(X)

            # logging performance
            self.eval_test.append(float(eval_func(torch.sigmoid(yHat).detach(), y)))
            if verbose:
                print(f'epoch {epoch} {eval_metric} - train: {np.mean(self.eval_train[epoch]):.3f} | test: {self.eval_test[-1]:.3f}')

            # early stop condition check
            if early_stopping(self.eval_test):
                break

    def predict(self, X: torch.tensor, sigmoid=True) -> np.array:
        return torch.sigmoid(self(X)).detach().numpy().reshape(-1) if sigmoid else self(X).detach().numpy().reshape(-1)
