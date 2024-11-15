import torch
from torch import nn
from typing import Dict


class ModelExample(nn.Module):
    def __init__(self, user_config: Dict):
        super().__init__()
        self.config = user_config
    def forward():
        pass
    