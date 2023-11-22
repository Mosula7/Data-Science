import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

class SimpleBinaryClassificationNN(nn.Module):
    
    @classmethod
    def layer_template(self):
        return print("""
                    {
                    'input': {'units':[None,None], 'batch_norm': None, 'dropout': None, 'activation': None},
                    'fcn'  : {'units':[None,None], 'batch_norm': None, 'dropout': None, 'activation': None},
                    'output':{'units':[None,None], 'batch_norm': None, 'dropout': None, 'activation': None}
                    }
                    """)
    
    
    def __init__(self,
                 layer_dict: dict,        
                 l1_lambda: float = None, # L1 regularization term
                 l2_lambda: float = None, # L1 regularization term
                 l1_not_applied: tuple = (),
                 l2_not_applied: tuple = (),
                 eval_metric: ('AUROC', 'Accuracy') = 'AUROC',
                 scaler = None) -> nn.Module:
        
        super(SimpleBinaryClassificationNN, self).__init__()
        
        self.layers = nn.ModuleDict()
        self.layer_names = [name for name in self.layers]

        self.l1_lambda = l1_lambda
        self.l1_not_applied = l1_not_applied

        self.l2_lambda = l2_lambda
        self.l2_not_applied = l2_not_applied

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

        self.scaler = scaler

        n_weights_l1 = 0
        for name,param in self.named_parameters():
            if not any(x in name for x in self.l1_not_applied):
                n_weights += param.numel()
        self.n_weights_l1 = n_weights_l1

        n_weights_l2 = 0
        for name,param in self.named_parameters():
            if not any(x in name for x in self.l2_not_applied):
                n_weights += param.numel()
        self.n_weights_l2 = n_weights_l2


    def forward(self, x):
        for layer in self.layer_names:
            x = self.layers[layer](x)
            return x 


    def l1_regularization(self): 
        # summing absolute values of all wights (except biases) to be used during gradient decent
        l1_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if not any(x in name for x in self.l1_not_applied):
                l1_regularization = l1_regularization + torch.sum(torch.abs(param))
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l1_regularization * self.l1_lambda / self.n_weights_l1
   

    def l2_regularization(self):
        # summing squared values of all wights (except biases) to be used during gradient decent 
        l2_regularization = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if not any(x in name for x in self.l2_not_applied):
                l2_regularization = l2_regularization + torch.sum(param ** 2) 
        # multiplying L2 loss by a scaling term and dividing by number of weights
        return l2_regularization * self.l2_lambda / self.n_weights_l2
    

    @classmethod
    def make_loader(self, X, y, batch_size, shuffle=False, scaler=None, fit_scaler=False):
        if type(X) != pd.DataFrame:
            return 'expected X to be a pandas DataFrame'
        X = X.fillna(0)

        if fit_scaler:
            X = scaler.fit_transform(X)
            data = TensorDataset(torch.tensor(X).float(), torch.tensor(y.values).reshape(-1,1).float())
            loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
            return loader, scaler

        elif scaler:
            X = scaler.transform(X)
            data = TensorDataset(torch.tensor(X).float(), torch.tensor(y.values).reshape(-1,1).float())
            loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
            return loader
        
    
    def fit(self, lossfun, optimizer, train_loader, test_loader, epochs, early_stop):

        self.eval_train = {}
        self.eval_test = []
        self.losses = {}
        
        no_imp = 0

        for epoch in range(epochs):
            self.train() # switching model back to training mode after evaluating it on the validation set
            # looping over all the batches

            self.eval_train[epoch] = []
            self.losses[epoch] = []

            for i, (X,y) in enumerate(train_loader):

                yHat = self(X) # forward pass
                loss = lossfun(yHat,y) + self.l1_regularization() + self.l2_regularization() # calculating loss
               
                # backpropagation
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()
               

                # evaluating the model on the batch and switching back to training mode
                self.eval()
                with torch.no_grad():
                    yHat = self(X)
                self.train()

                # logging batch performance
                self.eval_train[epoch].append(float(self.eval_func(y, torch.sigmoid(yHat).detach().numpy())))
                self.losses[epoch].append(loss.detach().item())
                print(f'batch {i} - {self.eval_metric}: {self.eval_train[epoch][-1]:.3f} | loss: {self.losses[epoch][-1]:.3f}', end = '\r')

        
            # evaluating on the validation set
            X,y = next(iter(test_loader))
            self.eval()
            with torch.no_grad():
                yHat = self(X)

            # logging performance
            self.eval_test.append(float(self.eval_func(y, torch.sigmoid(yHat).detach())))
            print(f'epoch {epoch} - train {self.eval_metric}: {torch.mean(self.eval_train[epoch]):.3f} | test {self.eval_metric}: {self.eval_test[-1]:.3f}')


            # resetting no improvement counter if the last iteration of the model had the best performance
            if self.eval_test and max(self.eval_test) == self.eval_test[-1]:
                no_imp = 0

            # early stop condition check

            if self.eval_test and max(self.eval_test) > self.eval_test[-1]:
                no_imp += 1
                if no_imp == early_stop:
                    print(f'no improvement for {early_stop} epochs, early stop')
                    break

    
    def predict_proba(self, X, scaled_tensor=False):

        if not scaled_tensor:
            X = self.scaler.transform(X.fillna(0))
            X = torch.tensor(X).float()

        if type(X) in (torch.utils.data.DataLoader, torch.tensor):
            self.eval()
            with torch.no_grad():
                return torch.sigmoid(self(X)).detach().numpy().reshape(-1)