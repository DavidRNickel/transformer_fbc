"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

class Config():
    def __init__(self):
        # settings for communications-related stuff
        self.K = 51 # length of bitstream
        self.M = 3 # length of shorter block
        assert(self.K % self.M == 0)
        self.N = self.M * self.K 
        self.knowledge_vec_len = self.M + 2*(self.N - 1) # all old bits and all feedback
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
        self.optim_lr = .0005
        self.optim_weight_decay = .01

        self.num_epochs = 100
        self.batch_size = 2500
        self.num_training_samps = int(1E6)
        self.num_valid_samps = int(1e5)
        assert(self.num_training_samps % self.batch_size == 0)
        assert(self.num_valid_samps % self.batch_size == 0)
        self.num_iters_per_epoch = self.num_training_samps // self.batch_size
        self.grad_clip = .5
        self.use_tensorboard = False
        self.use_belief_network = True