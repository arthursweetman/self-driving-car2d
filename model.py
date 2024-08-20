import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


torch.device('cuda')

class FFnn(nn.Module):
    def __init__(self):
        super(FFnn, self).__init__()
        self.ff1 = nn.Linear(68, 64)
        self.ff2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x))
        x = F.softmax(self.out(x), dim=0)
        _, pred = torch.max(x, dim=0)
        one_hot = F.one_hot(pred, num_classes=4)
        return one_hot


model = FFnn()

def step(x_input):
    x_input = torch.tensor(x_input)
    outputs = model(x_input)
    return outputs

if __name__ == "__main__":
    x_input = torch.randn(100)
    out = step(x_input)
    print(out)