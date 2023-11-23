import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchmetrics
import operator


class SimpleBinaryClassificationNN(nn.Module):
    def __init__(self,
                 layer_dict: dict,        
                 l1_lambda: float = None, # L1 regularization term
                 l2_lambda: float = None, # L2 regularization term
                 eval_metric: ('AUROC', 'Accuracy') = 'AUROC', # metric to log while training
                 ) -> nn.Module:
        
        super(SimpleBinaryClassificationNN, self).__init__()
        
        self.layers = nn.ModuleDict()

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.eval_metric = eval_metric
        self.eval_func = getattr(torchmetrics, eval_metric)(task='binary')

        self.layers = nn.ModuleDict()
        for key, values in layer_dict.items():
            self.layers[f'fc_{key}'] = nn.Linear(values['units'][0], values['units'][1])

            if values['batch_norm']:
                self.layers[f'bn_{key}'] = nn.BatchNorm1d(values['units'][1])
            if values['dropout']:
                self.layers[f'do_{key}'] = nn.Dropout(values['dropout'])
            if values['activation']:
                self.layers[f'act_{key}'] = getattr(torch.nn, values['activation'])()

        self.n_weights_l1 = 0
        for name,param in self.named_parameters():
            if ('weight' in name) and ('bn' not in name):
                self.n_weights_l1 += param.numel()
        
        self.n_weights_l2 = 0
        for name,param in self.named_parameters():
            if 'weight' in name:
                self.n_weights_l2 += param.numel()


    def l1_regularization(self): 
        # summing absolute values of all wights (except biases) to be used during gradient decent
        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if ('weight' in name) and ('bn' not in name):
                l1_regularization = l1_regularization + torch.sum(torch.abs(param))
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l1_regularization * self.l1_lambda / self.n_weights_l1
   

    def l2_regularization(self):
        # summing squared values of all wights (except biases) to be used during gradient decent 
        l2_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
           if ('weight' in name):
                l2_regularization = l2_regularization + torch.sum(torch.square(param)) 
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l2_regularization * self.l2_lambda / self.n_weights_l2


    def stop(self, objective):
        if objective == 'min':
            comp_fn = operator.lt
        elif objective == 'max':
            comp_fn = operator.gt
        # early stop condition check
        if self.eval_test and comp_fn(max(self.eval_test), self.eval_test[-1]):
            self.no_imp += 1
            if self.no_imp == self.early_stop:
                return True
        else:
            self.no_imp = 0
            return False


    def forward(self, x):
        for layer in self.layers:
            x = self.layers[layer](x)
        return x 
        
    
    def fit(self, lossfun, optimizer, train_loader, test_loader, epochs:int, early_stop: int, objective: ('min', 'max')):
        self.eval_train = {}
        self.eval_test = []
        self.losses = {}
        
        self.early_stop = early_stop
        self.no_imp = 0 # variable for tracking if the eval metric didn't improve 
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
                self.eval_train[epoch].append(float(self.eval_func(torch.sigmoid(yHat).detach(), y)))
                self.losses[epoch].append(loss.detach().item())
                print(f'batch {i} - {self.eval_metric}: {self.eval_train[epoch][-1]:.3f} | loss: {self.losses[epoch][-1]:.3f}', end = '\r')

        
            # evaluating on the test set
            X,y = next(iter(test_loader))
            self.eval()
            with torch.no_grad():
                yHat = self(X)

            # logging performance
            self.eval_test.append(float(self.eval_func(torch.sigmoid(yHat).detach(), y)))
            print(f'epoch {epoch} - train {self.eval_metric}: {np.mean(self.eval_train[epoch]):.3f} | test {self.eval_metric}: {self.eval_test[-1]:.3f}')

            # early stop condition check
            if self.stop(objective=objective):
                print(f'no improvement for {self.no_imp} rounds, early stop')
                break


    def predict_proba(self, X: torch.tensor) -> np.array:
        return torch.sigmoid(self(X)).detach().numpy().reshape(-1)
    
    
    def predict(self, X: torch.tensor) -> np.array:
        return self(X).detach().numpy().reshape(-1)
    
    
    @classmethod
    def layer_template(self) -> None:
        return "{ 'name': {'units':[None,None], 'batch_norm': None, 'dropout': None, 'activation': None} }"


    @classmethod
    def make_loader(self, 
                    X: pd.DataFrame, 
                    y: pd.Series, 
                    batch_size: int, 
                    shuffle: bool=False, 
                    fit_scaler: bool=False,
                    scaler=None) -> torch.utils.data.DataLoader:
        
        if type(X) != pd.DataFrame:
            return 'expected X to be a pd.DataFrame'
        X = X.fillna(0)

        if fit_scaler:
            scaler = scaler()
            X = scaler.fit_transform(X)
            data = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y.values).reshape(-1,1).float())
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
            return loader, scaler

        elif scaler:
            X = scaler.transform(X)
            data = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y.values).reshape(-1,1).float())
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
            return loader