"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

class Config():
    def __init__(self):
        self.use_tensorboard = True

        # settings for communications-related stuff
        self.K = 6 # length of bitstream
        self.M = 3 # length of shorter block
        assert(self.K % self.M == 0)
        self.T = 9
        self.N = self.T
        self.knowledge_vec_len = self.M + 2*(self.T - 1)
        self.snr_ff = 1 # in dB
        self.snr_fb = 20 # in dB
        self.noise_pwr_ff = 10**(-self.snr_ff/10)
        self.noise_pwr_fb = 10**(-self.snr_fb/10)

        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 20
        self.batch_size = 2500
        self.num_training_samps = int(1E6)
        self.num_valid_samps = int(1e5)
        assert(self.num_training_samps % self.batch_size == 0)
        assert(self.num_valid_samps % self.batch_size == 0)
        self.num_iters_per_epoch = self.num_training_samps // self.batch_size
        self.optim_lr = .01
        self.optim_weight_decay = .01
        self.grad_clip = .5