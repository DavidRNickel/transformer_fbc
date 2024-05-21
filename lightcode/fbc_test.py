import numpy as np
from math import sqrt, pi
from scipy.special import j0 #0th order Bessel function, generic softmax
import sys

import torch
from torch import nn
import torch.nn.functional as F

from timer_class import Timer

timer = Timer()

# constants
ONE_OVER_SQRT_TWO = 1/np.sqrt(2)
rng = np.random.default_rng()
fd = 10
T = 100E-3
RHO = j0(2*pi*fd*T)
SQRT_ONE_MIN_RHO_2 = sqrt(1 - RHO**2)


class FeedbackCode(nn.Module):
    def __init__(self, conf):
        super(FeedbackCode, self).__init__()

        self.conf = conf
        self.device = conf.device
        self.batch_size = conf.batch_size
        self.N = conf.N
        self.K = conf.K
        self.M = conf.M
        self.T = conf.T
        self.num_blocks = int(self.K // self.M)
        self.noise_pwr_ff = conf.noise_pwr_ff
        self.noise_pwr_fb = conf.noise_pwr_fb
        self.training = True
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.encoder = self.make_enc_dec_block(conf.knowledge_vec_len, 1, True)
        self.decoder = self.make_enc_dec_block(self.T + self.M, 2**self.M, False)

        # Power weighting-related parameters.
        self.weight_power = torch.nn.Parameter(torch.Tensor(self.T), requires_grad=True)
        self.weight_power.data.uniform_(1., 1.)
        self.weight_power_normalized = torch.sqrt(self.weight_power**2 * (self.T)/torch.sum(self.weight_power**2))
        self.transmit_power_tracking = []

        # Parameters for normalizing mean and variance of 
        self.mean_batch = torch.zeros(self.T)
        self.std_batch = torch.ones(self.T)
        self.mean_saved = torch.zeros(self.T)
        self.normalization_with_saved_data = False # True: inference w/ saved mean, var; False: calculate mean, var

    #
    # forward() calls both encoder and decoder. For evaluation, it is expected that the user
    # has provided forward & feedback noise 2D matrices, as well as a 
    def forward(self, bitstreams, noise_ff=None, noise_fb=None):
        knowledge_vecs = self.make_knowledge_vecs(bitstreams)
        self.weight_power_normalized = torch.sqrt(self.weight_power**2 * (self.T) / (self.weight_power**2).sum())
        if noise_ff is None:
            noise_ff = sqrt(self.noise_pwr_ff) * torch.randn((self.batch_size, self.T)).to(self.device)
            noise_fb = sqrt(self.noise_pwr_fb) * torch.randn((self.batch_size, self.T)).to(self.device)
        else:
            noise_ff = noise_ff
            noise_fb = noise_fb

        self.recvd_y = 0*torch.ones((self.batch_size, self.T)).to(self.device)
        self.transmit_power_tracking = []

        for t in range(self.T):
            print(t)
            x = self.transmit_bits_from_encoder(knowledge_vecs, t)

            y_tilde = self.process_bits_at_receiver(x, t, noise_ff, noise_fb)

            if t!=0:
                self.recvd_y_tilde = torch.hstack((self.recvd_y_tilde,y_tilde.unsqueeze(-1)))
                self.prev_xmit_signal = torch.hstack((self.prev_xmit_signal, x.unsqueeze(-1)))
            else:
                self.prev_xmit_signal = x.unsqueeze(-1)
                self.recvd_y_tilde = y_tilde.unsqueeze(-1)
            
            knowledge_vecs = self.make_knowledge_vecs(bitstreams,
                                                      fb_info=self.recvd_y_tilde, 
                                                      prev_x=self.prev_xmit_signal)

        dec_out = self.decode_received_symbols(torch.hstack((self.recvd_y, bitstreams)))

        return dec_out

    #
    #
    def make_knowledge_vecs(self, b, fb_info=None, prev_x=None):
        if fb_info is None:
            fbi = 0 * torch.ones(self.batch_size, self.T - 1).to(self.device)
            px = 0 * torch.ones(self.batch_size, self.T - 1).to(self.device)
            q = torch.hstack((px, fbi))
        else:
            q = torch.hstack((prev_x, fb_info))
            q = F.pad(q, pad=(0, 2*(self.T - 1) - q.shape[-1]), value=0)

        return torch.hstack((b, q))
    
    #
    # Do all the transmissions from the encoder side to the decoder side.
    def transmit_bits_from_encoder(self, k, t):
        x = self.encoder(k)
        return self.normalize_transmit_signal_power(x, t).squeeze(-1)

    #
    # Process the received symbols at the decoder side. NOT THE DECODING STEP!!!
    def process_bits_at_receiver(self, x, t, noise_ff, noise_fb):
        self.transmit_power_tracking.append(torch.sum(torch.abs(x)**2).detach().clone().cpu().numpy())
        y =  x + noise_ff[:,t]
        self.recvd_y[:,t] = y
        y_tilde = y + noise_fb[:,t]

        return y_tilde

    #
    # Actually decode all of the received symbols.
    def decode_received_symbols(self,y):
        y = self.decoder(y)

        return y

    #
    # Take the input bitstreams and map them to their one-hot representation.
    def bits_to_one_hot(self, bitstreams):
        # This is a torch adaptation of https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
        # It maps binary representations to their one-hot values by first converting the rows into 
        # the base-10 representation of the binary.
        x = (bitstreams * (1<<torch.arange(bitstreams.shape[-1]-1,-1,-1).to(self.device))).sum(1)

        return F.one_hot(x, num_classes=2**self.M)

    #
    # Map the onehot representations into their binary representations.
    def one_hot_to_bits(self, onehots):
        x = torch.argmax(onehots,dim=1)
        # Adapted from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        bin_representations = (((x[:,None] & (1 << torch.arange(self.M,requires_grad=False).to(self.device).flip(0)))) > 0).int()

        return bin_representations

    #
    #
    def calc_error_rates(self, bit_estimates, bits):
        if not isinstance(bits,np.ndarray):
            not_eq = torch.not_equal(bit_estimates, bits)
            ber = not_eq.float().mean()
            bler = (not_eq.sum(1)>0).float().mean()
        else:
            not_eq = np.not_equal(bit_estimates, bits)
            ber = not_eq.mean()
            bler = (not_eq.sum(1)>0).mean()

        return ber, bler

    #
    # The following methods are from https://anonymous.4open.science/r/RCode1/main_RobustFeedbackCoding.ipynb
    # which is the code for the paper "Robust Non-Linear Feedback Coding via Power-Constrained Deep Learning".
    #

    #
    # Handle the power weighting on the transmit bits.
    def normalize_transmit_signal_power(self, x, t):
        x = self.normalization(x, t)

        return self.weight_power_normalized[t] * x 

    #
    # Normalize the batch.
    def normalization(self, inputs, t_idx):
        mean_batch = torch.mean(inputs)
        std_batch = torch.std(inputs)
        if self.training == True:
            outputs = (inputs - mean_batch) / std_batch
        else:
            if self.normalization_with_saved_data:
                outputs = (inputs - self.mean_saved[t_idx]) / self.std_saved[t_idx]
            else:
                self.mean_batch[t_idx] = mean_batch
                self.std_batch[t_idx] = std_batch
                outputs = (inputs - mean_batch) / std_batch

        return outputs
    
    #
    #
    def make_enc_dec_block(self, dim_in, dim_out, is_enc):
        lin1 = nn.Linear(dim_in, 32).to(self.device)
        lin2 = nn.Linear(32,32).to(self.device)
        lin3 = nn.Linear(32,32).to(self.device)
        lin4 = nn.Linear(64,16).to(self.device)
        mlp = self._make_mlp(16, dim_out, 2 if is_enc else 1)
        self._init_weights((lin1,lin2,lin3,lin4))

        def _fe(y):
            x1 = lin1(y)
            x = self.relu(x1)
            x = lin2(x)
            x = self.relu(x)
            x = lin3(x)
            x = lin4(torch.hstack((x,x1)))
            x = mlp(x)
            return x
        
        return _fe
    
    #
    #
    def _make_mlp(self, dim_in, dim_out, num_layers):
        if num_layers==1:
            lin1 = nn.Linear(dim_in, dim_out).to(self.device)
            self._init_weights((lin1,))
            def _mlp(x):
                x = lin1(x)
                return x
        else:
            lin1 = nn.Linear(dim_in, 32).to(self.device)
            lin2 = nn.Linear(32, dim_out).to(self.device)
            self._init_weights((lin1,lin2))
            def _mlp(x):
                x = lin1(x)
                x = self.relu(x)
                x = lin2(x)
                return x

        return _mlp

    #
    #
    def _init_weights(self, layers):
        for layer in layers:
            torch.nn.init.kaiming_uniform_(layer.weight)