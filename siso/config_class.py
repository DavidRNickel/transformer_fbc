"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

class Config():
    def __init__(self):
        # settings for communications-related stuff
        self.N = 6 # number channel usees
        self.K = 3 # length of bitstream
        # +1 for start of sequence token, -1 since we only need N-1 feedback info slots.
        # self.knowledge_vec_len = 1 + self.K + self.N - 1
        self.knowledge_vec_len = self.K + self.N - 1
        self.snr_ff = 1 # in dB
        self.snr_fb = 20 # in dB
        self.noise_pwr_ff = 10**(-self.snr_ff/10)
        self.noise_pwr_fb = 10**(-self.snr_fb/10)

        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')
        self.max_len_enc = self.N
        self.num_layers_xmit = 2 
        self.num_layers_recv = 3
        self.n_heads = 1
        self.d_model = 32 
        self.scaling_factor = 4
        self.dropout = 0.0

        self.num_epochs = 10
        self.batch_size = 1000
        self.num_training_samps = int(100000)
        self.num_iters_per_epoch = self.num_training_samps // self.batch_size
        self.grad_clip = 1
        self.num_valid_samps = int(100000)
        self.pooling_type = 'first'

        self.use_tensorboard = True
