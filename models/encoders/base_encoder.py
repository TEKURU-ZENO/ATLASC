import torch.nn as nn

class BaseEncoder(nn.Module):
    def forward(self, x):
        raise NotImplementedError
