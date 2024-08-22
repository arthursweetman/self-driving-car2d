import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


torch.device('cuda')
dropout = 0.2

class Agent():
    def __init__(self):

        self.model = FFnn(68)
        self.target_model = FFnn(68)

    def step(self, x_input):
        x_input = torch.tensor(x_input)
        Q_values = self.model(x_input)
        Q_values_target = self.target_model(x_input)
        return Q_values, Q_values_target

    def reward(self):
        pass


class FFnn(nn.Module):
    def __init__(self, input_dim):
        super(FFnn, self).__init__()
        self.input_dim = input_dim
        self.ff1 = nn.Linear(input_dim, 256)
        self.ff2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        l1 = self.ff1(x)
        out1 = torch.relu(l1)
        out1 = self.dropout(out1)

        l2 = self.ff3(out1)  # Linear activation
        out3 = self.dropout(l2)

        _, pred = torch.max(out3, dim=0)
        one_hot = F.one_hot(pred, num_classes=4)

        return one_hot

    def loss(self):
        pass


model = FFnn(68)

def step(x_input):
    x_input = torch.tensor(x_input)
    outputs = model(x_input)
    return outputs

if __name__ == "__main__":
    x_input = torch.randn(100)
    out = step(x_input)
    print(out)