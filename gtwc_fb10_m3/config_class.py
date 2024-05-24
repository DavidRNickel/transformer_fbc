import torch

class Config():
    def __init__(self):
        self.use_tensorboard = False
        self.use_belief_network = True
        self.loadfile = None

        # settings for communications-related stuff
        self.K = 6 # length of bitstream
        self.M = 3 # length of shorter block
        assert(self.K % self.M == 0)
        self.T = 9
        self.N = self.M * self.K 
        self.knowledge_vec_len = self.M + 2*(self.T - 1) 
        if self.use_belief_network:
            self.knowledge_vec_len += 2*self.M
        self.snr_ff = 1 # in dB
        self.snr_fb = 10 # in dB
        self.noise_pwr_ff = 10**(-self.snr_ff/10)
        self.noise_pwr_fb = 10**(-self.snr_fb/10)

        # Model settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')
        self.max_len_enc = self.N
        self.num_layers_xmit = 2 
        self.num_layers_belief = 2
        self.num_layers_recv = 3
        self.n_heads = 1
        self.d_model = 32
        self.scaling_factor = 4
        self.dropout = 0.0

        self.num_epochs = 50
        self.batch_size = 25000
        self.num_training_samps = int(1E7)
        self.num_valid_samps = int(1E6)
        self.num_test_samps = int(5E5)
        self.num_infer_samps = int(1E8)
        assert(self.num_training_samps % self.batch_size == 0)
        assert(self.num_valid_samps % self.batch_size == 0)
        assert(self.num_test_samps % self.batch_size == 0)
        assert(self.num_infer_samps % self.batch_size == 0)
        self.num_iters_per_epoch = self.num_training_samps // self.batch_size
        self.optim_lr = .001
        self.optim_weight_decay = .01
        self.grad_clip = 1
