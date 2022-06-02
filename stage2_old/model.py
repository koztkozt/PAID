import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math

#  In the prototype of RAIDS, we build a classifier that is mainly composed of two linear layers.
class stage2(nn.Module):
    def __init__(self):
        super(stage2, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        out = self.fc1(x)  # Concatenates a sequence of tensors along a new dimension. torch.stack(inp)
        out = self.fc2(out)
        out = F.softmax(out, dim=-1)
        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)