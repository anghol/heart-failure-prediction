import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class HeartDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        
        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        
        self.length = len(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length


class NeuralNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def train(model: nn.Module,
          num_epochs: int,
          train_loader,
          optimizer,
          loss_fn,
          valid_loader,
          epoch_logging_interval=10):
    
    log_dict = {'losses_per_epoch': [], 'accuracy_per_epoch': []}
    
    for epoch in range(num_epochs):

        model.train()
        loss_curr_epoch = 0
        for batch, (x, y) in enumerate(train_loader):
            y = y.view(-1, 1)
            y_hat = model(x)

            # Усреднённый лосс по батчам
            loss = loss_fn(y_hat, y)
            loss_curr_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        model.eval()
        accuracy_curr_epoch = 0
        for batch, (x, y) in enumerate(valid_loader):
            y_pred = model(x).view(-1)

            # Среднее accuracy по батчам
            accuracy_curr_epoch += torch.eq(y_pred.round(), y).detach().numpy().mean()

        log_dict['losses_per_epoch'].append(loss_curr_epoch / len(train_loader))
        log_dict['accuracy_per_epoch'].append(accuracy_curr_epoch / len(valid_loader))

        if (epoch + 1) % epoch_logging_interval == 0:
            print(f'Epoch {epoch + 1}:\tLoss {log_dict["losses_per_epoch"][epoch]}, Accuracy {log_dict["accuracy_per_epoch"][epoch]}')

    return log_dict