import torch

class Config():
    def __init__(self):
        self.use_tensorboard = False

        # settings for communications-related stuff
        self.K = 3 # length of bitstream
        self.M = 3 # length of shorter block
        assert(self.K % self.M == 0)
        self.N = self.M * self.K 
        self.T = 9
        self.knowledge_vec_len = self.M + 2*(self.T - 1) # all old bits and all feedback
        self.snr_ff = 20 # in dB
        self.snr_fb = 20 # in dB
        self.noise_pwr_ff = 10**(-self.snr_ff/10)
        self.noise_pwr_fb = 10**(-self.snr_fb/10)

        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optim_lr = .001
        self.optim_weight_decay = .01
        self.num_epochs = 120
        self.batch_size = int(5E5)
        self.num_training_samps = int(1E7)
        self.num_valid_samps = int(1e6)
        assert(self.num_training_samps % self.batch_size == 0)
        assert(self.num_valid_samps % self.batch_size == 0)
        self.num_iters_per_epoch = self.num_training_samps // self.batch_size
        self.grad_clip = .5