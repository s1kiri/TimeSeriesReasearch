import torch
from torch import nn
from abc import ABC
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import logging
import logging.config
import yaml
from models.rnn import LSTMModel
from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class BaseConfig:
    device: str = field(default='cpu')
    batch_size: int = field(default=16)
    eval_batch_size: int = field(default=1)
    num_epochs: int = field(default=3)
    learning_rate: float = field(default=0.0003)
    adam_betas: Tuple = field(default=(0.95, 0.99))
    test_size_percentage: float = field(default=0.05)
    predictor_period: int = field(default=30)
    forecast_period: int = field(default=1)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor, base_config: BaseConfig):
        self.base_config = base_config
        self.data, self.target = torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.base_config.predictor_period

    def __getitem__(self, idx):
        X_seq = self.data[idx:idx + self.base_config.predictor_period]
        y_seq = self.target[idx + self.base_config.predictor_period]

        return X_seq, y_seq

class BaseNetwork():
    def __init__(
            self, config_path: str,
            model: torch.nn.Module,
            data: torch.Tensor,
            target: torch.Tensor
        ):
        super().__init__()
        config = self.read_config(config_path)
        logging_config = self.read_config('configs/logging_config.yaml')
        if logging_config:
            logging.config.dictConfig(logging_config)
            print("Logging configuration loaded successfully")
        else:
            print("Failed to load logging configuration")

        self.logger = logging.getLogger('BaseNetwork')
        self.logger.info("Logger successfully initialized!")

        self.base_config = BaseConfig.from_dict(config['BaseConfig'])
        self.user_config = config['UserConfig']

        self.model = model
        
        self.model.to(self.base_config.device)

        self.optimizer = Adam(self.model.parameters(), 
                          lr=self.base_config.learning_rate)
        
        self.criterion = nn.MSELoss()
        
        self.train_dataset, self.eval_dataset = self.create_datasets(data, target)
    
    def create_datasets(self, X, y):
        test_size_percentage = self.base_config.test_size_percentage
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= test_size_percentage, shuffle=False)

        return TimeSeriesDataset(X_train, y_train, self.base_config), TimeSeriesDataset(X_test, y_test, self.base_config)
    
    def pipeline(self):

        train_loader = DataLoader(self.train_dataset, 
                            batch_size=self.base_config.batch_size,
                            shuffle = False)
        eval_loader = DataLoader(self.eval_dataset, 
                                 batch_size=self.base_config.eval_batch_size, 
                                 shuffle=False)


        model = self.train_model(train_loader, False, False)

        test = self.evaluate(eval_loader)

        return test
    
    def train_model(self, train_loader, return_model: bool, return_loss_logs: bool):
        self.logger.info('Model has been initialized')

        loss_logs = list()
        self.logger.info('Optimizer and loss have been initialized')
        for epoch in range(self.base_config.num_epochs):
            self.model.train()
            total_loss = 0
            self.logger.info(f'Epoch {epoch}')
            for X_batch, y_batch in (pbar := tqdm(train_loader)):
                X_batch, y_batch = X_batch.to(self.base_config.device), y_batch.to(self.base_config.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

                pbar.set_description(f'Training loss: {loss.item()}')
                loss_logs.append(loss.item())

            self.logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
        if return_model and return_loss_logs:
            return self.model, loss_logs
        elif return_model:
            return self.model
        elif return_loss_logs:
            return loss_logs
    
    def evaluate(self, eval_loader):
        all_preds = []
        all_true = []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in eval_loader:
                X_batch, y_batch = X_batch.to(self.base_config.device), y_batch.to(self.base_config.device)
                y_pred = self.model(X_batch)
                all_preds.append(y_pred.cpu().numpy())
                all_true.append(y_batch.cpu().numpy())

        return all_preds, all_true

        
    @staticmethod
    def read_config(path: str) -> dict:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config



