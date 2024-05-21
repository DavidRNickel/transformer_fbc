import torch
from fbc_test import *
from config_class import *

if __name__=='__main__':
    device = 'cpu'
    conf = Config()
    fbc = FeedbackCode(conf)
    x = torch.randint(0,2,(3,4,5)).int().to(device)