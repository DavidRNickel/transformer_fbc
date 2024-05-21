import torch
from torch import nn

if __name__=='__main__':
    def fe(dim_in, dim_out):
        lin1 = nn.Linear(dim_in,32)
        lin2 = nn.Linear(32,dim_out)
        relu = nn.ReLU()
        def _fe(x):
            x = lin1(x)
            x = relu(x)
            x = lin2(x)
            x = relu(x)
            return x
        return _fe

    feature_extractor = fe(4,20)
    x = torch.randint(0,10,(3,5,4)).float()
    print(feature_extractor(x).shape)