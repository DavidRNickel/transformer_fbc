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
        self.K = 6 # length of bitstream
        self.num_xmit_chans = 2
        self.knowledge_vec_len = self.K + 2*self.num_xmit_chans + self.N - 1
        self.snr_ff = -1 # in dB
        self.snr_fb = -1 # in dB
        self.noise_pwr_ff = 10**(-self.snr_ff/10)
        self.noise_pwr_fb = 10**(-self.snr_fb/10)

        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 3# set to 8192
        self.max_len_enc = self.N
        self.num_layers_xmit = 2 
        self.num_layers_recv = 3
        self.n_heads = 1
        self.d_model = 32 # self.n_heads * self.knowledge_vec_len
        self.scaling_factor = 4
        self.dropout = 0.0

        self.num_epochs = 100
        self.grad_clip = 1

        self.use_tensorboard = True