import torch
from torch import nn


if __name__=='__main__':
    embed = nn.Sequential(nn.Linear(1,32),
                          nn.Tanh())
    
    x = torch.ones((10, 5, 1))

    print(embed(x).shape)