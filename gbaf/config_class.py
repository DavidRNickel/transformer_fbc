import torch

class Config():
    def __init__(self):
        self.use_tensorboard = False
        self.use_belief_network = True
        self.loadfile = None

        # settings for communications-related stuff
        self.K = 51 # length of bitstream
        self.M = 3 # length of shorter block
        assert(self.K % self.M == 0)
        self.N = self.M * self.K 
        self.T = 9
        self.knowledge_vec_len = self.M + 2*(self.T - 1) # all old bits and all feedback
        if self.use_belief_network:
            self.knowledge_vec_len += 2*self.M
        self.snr_ff = 1 # in dB
        self.snr_fb = 20 # in dB
        self.noise_pwr_ff = 10**(-self.snr_ff/10)
        self.noise_pwr_fb = 10**(-self.snr_fb/10)

        # Model settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {self.device}')
        self.max_len_enc = self.N
        self.num_layers_xmit = 2 
        self.num_layers_belief = 2
        self.num_layers_recv = 3
        self.n_heads = 1
        self.d_model = 32
        self.scaling_factor = 4
        self.dropout = 0.0
        self.optim_lr = .001
        self.optim_weight_decay = .01

        self.num_epochs = int(1E5)
        self.batch_size = 8192
        self.num_valid_epochs = 125
        self.grad_clip = .5
        self.save_freq = int(2500)
        self.print_freq = int(500)
