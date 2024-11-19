import torch
import torch.nn as nn
from typing import Dict

# class RNNModel(nn.Module):
#     def __init__(self, user_config: Dict):
#         super().__init__()
#         self.user_config = user_config
#         self.rnn = nn.RNN(self.user_config['input_size'], self.user_config['hidden_size'], self.user_config['num_layers'], batch_first = self.user_config['batch_first'])
#         self.fc = nn.Linear(self.user_config['hidden_size'], 1)

#     def forward(self, x):
#         h0 = torch.zeros(self.user_config['num_layers'], x.size(0), self.user_config['hidden_size']).to(x.device)
#         out, _ = self.rnn(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out.unsqueeze(1)

class RNNModel(nn.Module):
    def __init__(self, user_config: Dict):
        super(RNNModel, self).__init__()
        self.user_config = user_config
        self.rnn = nn.RNN(input_size=self.user_config['input_size'], 
                          hidden_size=self.user_config['hidden_size'], 
                          num_layers=self.user_config['num_layers'], 
                          batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(self.user_config['hidden_size'], self.user_config['hidden_size'] * 2),
            nn.ReLU(),
            nn.Linear(self.user_config['hidden_size'] * 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов для всех слоёв."""
        # Инициализация весов RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier для весов
            elif 'bias' in name:
                nn.init.zeros_(param)  # Нули для смещений

        # Инициализация весов внутри fc (Sequential)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):  # Если это Linear слой
                nn.init.xavier_uniform_(layer.weight)  # Xavier для весов
                nn.init.zeros_(layer.bias)  # Нули для смещений

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        _, hidden = self.rnn(x)  # hidden: (num_layers, batch_size, hidden_size)
        hidden = hidden[-1]  # Используем скрытое состояние последнего слоя
        output = self.fc(hidden)  # output: (batch_size, output_size)
        return output

class LSTM(nn.Module):
    def __init__(self, user_config: Dict):
        super().__init__()
        self.user_config = user_config
        self.lstm = nn.LSTM(
            input_size=self.user_config['input_size'], 
            hidden_size=self.user_config['hidden_size'], 
            num_layers=self.user_config['num_layers'], 
            batch_first=True, 
            dropout=0.2
        )
        
        self.fc = nn.Linear(self.user_config['hidden_size'], 1)

        # self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param[n // 4:n // 2].data.fill_(1.0)  # Смещение для забывающего слоя

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        output = self.fc(last_hidden)
        return output
class LSTMModel(nn.Module):
    def __init__(self,  user_config:Dict):
        super(LSTMModel, self).__init__()
        self.user_config = user_config
        self.lstm = nn.LSTM(input_size=self.user_config['input_size'], 
                            hidden_size=self.user_config['hidden_size'],
                            num_layers=self.user_config['num_layers'],
                            batch_first=True)
        self.fc = nn.Linear(self.user_config['hidden_size'], 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # берем последний временной шаг
        return out

    
class MLP(nn.Module):
    def __init__(self, user_config: Dict):
        super().__init__()
        
        self.user_config = user_config

        self.fc = nn.Sequential(
            nn.Linear(self.user_config['input_size'], self.user_config['hidden_size'] * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.user_config['hidden_size'] * 4, self.user_config['hidden_size'] * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.user_config['hidden_size'] * 2, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):  # Если это Linear слой
                nn.init.xavier_uniform_(layer.weight)  # Xavier для весов
                nn.init.zeros_(layer.bias)  # Нули для смещений
    def forward(self, x):
        out = self.fc(x)
        return out.mean(dim=1, keepdim=True)

