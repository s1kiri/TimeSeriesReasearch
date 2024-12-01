import torch
from torch import nn
from abc import ABC
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import logging
import logging.config
import yaml
from models.rnn import LSTM
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

class BaseNetwork():
    def __init__(
            self, config_path: str,
            model_class_constructor: torch.nn.Module, # it has to be a constructor, not a class object
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

        self.base_config = BaseConfig.from_dict(config['BaseConfig']) # creating a dataclass for base config
        self.user_config = config['UserConfig']

        self.model = LSTM(self.user_config)
        self.criterion = nn.MSELoss()
        
        self.dataset = TimeSeriesDataset # it is a constructor, to use for the first time call it with brackets, like this: self.dataset(data=data, target=target)
        # assert data.shape[-1] == target.shape[-1]
        self.train_data, self.test_data = self.data_spliter(data)
        self.train_target, self.test_target= self.data_spliter(target)

    def data_spliter(self, tensor: torch.Tensor):
        test_size_percentage = self.base_config.test_size_percentage
        train, test = train_test_split(tensor, test_size= test_size_percentage, shuffle=False)
        return train, test

    def evaluate(self, model):
        eval_dataset = self.dataset(data=self.test_data, 
                                    target=self.test_target, 
                                    base_config=self.base_config)
        eval_loader = DataLoader(eval_dataset, 
                                 batch_size=self.base_config.eval_batch_size, 
                                 shuffle=False)
        all_preds = []
        all_true = []
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for X, y in eval_loader:
                X, y = X.to(self.base_config.device), y.to(self.base_config.device)
                output = model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_true.append(y.cpu().numpy())
            self.logger.info(f'Validation Loss: {total_loss / len(eval_loader)}')
        return all_preds, all_true

    def train(self, return_model: bool, return_loss_logs: bool):
        train_dataset = self.dataset(data=self.train_data, 
                                    target=self.train_target, 
                                    base_config=self.base_config
                                    )
        
        loader = DataLoader(train_dataset, 
                            batch_size=self.base_config.batch_size,
                            shuffle = False)

        # model = self.model(self.user_config)
        self.model.to(self.base_config.device)
        self.logger.info('Model has been initialized')

        optimizer = AdamW(self.model.parameters(), 
                          lr=self.base_config.learning_rate,
                          betas=self.base_config.adam_betas,
                          weight_decay=0.1)

        loss_logs = list()
        self.logger.info('Optimizer and loss have been initialized')
        self.model.train()
        for epoch in range(self.base_config.num_epochs):
            total_loss = 0
            self.logger.info(f'Epoch {epoch}')
            for (X, y) in (pbar := tqdm(loader)):
                X, y = X.to(self.base_config.device), y.to(self.base_config.device)
                
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_description(f'Training loss: {loss.item()}')
                loss_logs.append(loss.item())

            self.logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')
        if return_model and return_loss_logs:
            return self.model, loss_logs
        elif return_model:
            return self.model
        elif return_loss_logs:
            return loss_logs
        
    @staticmethod
    def read_config(path: str) -> dict:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config


class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor, base_config: BaseConfig):
        self.base_config = base_config
        self.data, self.target = torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.base_config.predictor_period - self.base_config.forecast_period

    def __getitem__(self, idx):
        X_seq = self.data[idx:idx + self.base_config.predictor_period]
        y_seq = self.target[idx + self.base_config.predictor_period]

        return X_seq, y_seq
