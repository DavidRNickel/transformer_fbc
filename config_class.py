"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

class Config():
    def __init__(self):
        # settings for communications-related stuff
        self.N = 18 # number channel usees
        self.K = 7 # length of bitstream
        self.num_xmit_chans = 2
        self.knowledge_vec_len = self.K + 4*self.num_xmit_chans + self.N - 1

        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.max_len_enc = self.N
        self.n_layers = 2 
        self.n_heads = 8
        self.d_model = 32 #self.n_heads * self.knowledge_vec_len
        self.ffn_hidden = 256 # default 2048 
        self.drop_prob = 0.0

        # optimizer parameter setting
        self.init_lr = 1e-5
        self.factor = 0.9
        self.adam_eps = 5e-9
        self.patience = 10
        self.warmup = 100
        self.epoch = 1000
        self.clip = 1.0
        self.weight_decay = 5e-4
        self.inf = float('inf')
