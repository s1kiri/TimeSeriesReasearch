import torch
import torch.nn as nn
from typing import Dict

class LSTM(nn.Module):
    
    def __init__(self, user_config: Dict):
        super().__init__()
        self.user_config = user_config
        self.lstm = nn.LSTM(self.user_config['input_size'], self.user_config['hidden_size'], self.user_config['num_layers'], batch_first = self.user_config['batch_first'])
        self.fc = nn.Linear(self.user_config['hidden_size'], 1)

    def forward(self, x):
        h = torch.zeros(self.user_config['num_layers'], x.size(0), self.user_config['hidden_size'])
        c = torch.zeros(self.user_config['num_layers'], x.size(0), self.user_config['hidden_size'])
        out, _ = self.lstm(x, (h,c))
        out = self.fc(out[:, -1,:])
        return out
