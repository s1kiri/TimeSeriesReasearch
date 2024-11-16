import torch
from torch import nn
from abc import ABC
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import logging
import yaml
from tqdm import tqdm
from dataclasses import dataclass, field


class BaseNetwork():
    def __init__(
            self, config_path: str,
            model_class_constructor: torch.nn.Module, # it has to be a constructor, not a class object
            data: torch.Tensor,
            target: torch.Tensor
        ):
        super().__init__()
        config = self.read_config(config_path)
        self.base_config = BaseConfig.from_dict(config['BaseConfig']) # creating a dataclass for base config
        self.user_config = config['UserConfig']
        self.model = model_class_constructor
        self.logger = logging.getLogger(model_class_constructor.__name__)
        self.dataset = TimeSeriesDataset # it is a constructor, to use for the first time call it with brackets, like this: self.dataset(data=data, target=target)
        self.collator = TimeSeriesCollator # it is a constructor, to use for the first time call it with brackets, like this: self.collator(dataset, batch_size=batch_size, collate_fn=collator)
        assert data.shape[-1] == target.shape[-1]
        self.train_data, self.test_data = self.data_spliter(data)
        self.train_target, self.test_target= self.data_spliter(target)

    def data_spliter(self, tensor: torch.Tensor):
        test_size_percentage = self.base_config.test_size_percentage
        test_size = int(len(tensor) * test_size_percentage)
        return tensor[:-test_size,:], tensor[-test_size:,:]
    
    def evaluate(self, model):
        eval_dataset = self.dataset(data=self.test_data, 
                                    target=self.test_target, 
                                    base_config=self.base_config)
        batch_size = self.base_config.batch_size
        collator = self.collator(batch_size)
        eval_loader = DataLoader(eval_dataset, 
                                 batch_size=batch_size, 
                                 collate_fn=collator)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (X, y) in enumerate(eval_loader):
                X, y = X.to(self.base_config.device), y.to(self.base_config.device)
                outputs = model(X)
                loss = nn.MSELoss()(outputs, y)
                total_loss += loss.item()
            self.logger.info(f'Validation Loss: {total_loss / len(eval_loader)}')

    def train(self, return_model: bool, return_loss_logs: bool):
        dataset = self.dataset(data=self.train_data, 
                               target=self.train_target, 
                               base_config=self.base_config)
        batch_size = self.base_config.batch_size
        collator = self.collator(batch_size)
        loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=collator)

        model = self.model(self.user_config)
        model.to(self.base_config.device)
        self.logger.info('Model has been initialized')

        optimizer = Adam(model.parameters(), lr=self.base_config.learning_rate)
        criterion = nn.MSELoss()
        loss_logs = list()
        self.logger.info('Optimizer and loss have been initialized')
        
        for epoch in range(self.base_config.num_epochs):
            model.train()
            total_loss = 0
            self.logger.info(f'Epoch {epoch}')
            for (X, y) in (pbar := tqdm(loader)):
                X, y = X.to(self.base_config.device), y.to(self.base_config.device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_description(f'Training loss: {loss.item()}')
                loss_logs.append(loss.item())
            self.logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')
            self.logger.info('Validating...')
            self.evaluate(model)
        
        if return_model and return_loss_logs:
            return model, loss_logs
        elif return_model:
            return model
        elif return_loss_logs:
            return loss_logs
        
    @staticmethod
    def read_config(path: str) -> dict:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config


@dataclass
class BaseConfig:
    device: str = field(default='cpu')
    batch_size: int = field(default=16)
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
        self.data, self.target = self.dataslice(data=data, target=target)
        
    def dataslice(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictor_days = self.base_config.predictor_period
        forecast_days = self.base_config.forecast_period

        X_sequences = []
        y_sequences = []

        for i in range(len(data) - predictor_days - forecast_days + 1):
            predictors = data[i:i + predictor_days]
            target_seq = target[i + predictor_days:i + predictor_days + forecast_days]
            X_sequences.append(predictors)
            y_sequences.append(target_seq)
        return torch.stack(X_sequences), torch.stack(y_sequences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class TimeSeriesCollator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, batch):
        X_batch = []
        y_batch = []
        for X, y in batch:
            X_batch.append(X)
            y_batch.append(y) 
        X_batch = torch.stack(X_batch)
        y_batch = torch.stack(y_batch)
        return X_batch, y_batch
