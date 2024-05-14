import numpy as np
from math import sqrt, pi
from scipy.special import j0 #0th order Bessel function, generic softmax
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from pos_enc_test import PositionalEncoding
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
        self.d_model = conf.d_model
        self.device = conf.device
        self.batch_size = conf.batch_size
        self.N = conf.N
        self.K = conf.K
        self.M = conf.M
        self.T = int(self.M * self.N // self.K)
        self.num_blocks = int(self.K // self.M)
        self.noise_pwr_ff = conf.noise_pwr_ff
        self.noise_pwr_fb = conf.noise_pwr_fb
        self.training = True
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Set up the transmit side encoder.
        self.embedding_encoder = nn.Sequential(nn.Linear(conf.knowledge_vec_len, 96), 
                                               self.relu, 
                                               nn.Linear(96,96),
                                               self.relu,
                                               nn.Linear(96,self.d_model))
        self.pos_encoding_encoder = PositionalEncoding(d_model=conf.d_model, 
                                                       dropout=conf.dropout, 
                                                       max_len=self.num_blocks)
        self.enc_layer = TransformerEncoderLayer(d_model=conf.d_model, 
                                                 nhead=conf.n_heads, 
                                                 norm_first=True, 
                                                 dropout=conf.dropout, 
                                                 dim_feedforward=conf.scaling_factor*conf.d_model,
                                                 activation=self.relu,
                                                 batch_first=True)
        self.encoder = TransformerEncoder(self.enc_layer, num_layers=conf.num_layers_xmit)
        for name, param in self.encoder.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)
        self.enc_raw_output = nn.Linear(self.d_model, 1)
        nn.init.kaiming_uniform_(self.enc_raw_output.weight)

        # Set up the receive side decoder.
        self.embedding_decoder = nn.Sequential(nn.Linear(self.T, 96),
                                               self.relu, 
                                               nn.Linear(96,96),
                                               self.relu,
                                               nn.Linear(96,conf.d_model))
        self.pos_encoding_decoder = PositionalEncoding(d_model=conf.d_model, 
                                                       dropout=conf.dropout, 
                                                       max_len=self.num_blocks)
        self.dec_layer = TransformerEncoderLayer(d_model=conf.d_model,
                                                 nhead=conf.n_heads,
                                                 norm_first=True,
                                                 dropout=conf.dropout,
                                                 activation=self.relu,
                                                 batch_first=True)
        self.decoder = TransformerEncoder(self.dec_layer, num_layers=conf.num_layers_recv)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)
        self.dec_raw_output = nn.Linear(self.d_model, 2**self.M)
        nn.init.kaiming_uniform_(self.dec_raw_output.weight)

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
            noise_ff = sqrt(self.noise_pwr_ff) * torch.randn((self.batch_size, self.num_blocks, self.T)).to(self.device)
            noise_fb = sqrt(self.noise_pwr_fb) * torch.randn((self.batch_size, self.num_blocks, self.T)).to(self.device)
        else:
            noise_ff = noise_ff
            noise_fb = noise_fb
        self.recvd_y = torch.empty((self.batch_size, self.num_blocks, self.T)).to(self.device)
        self.transmit_power_tracking = []

        for t in range(self.T):
            # Transmit side
            x = self.transmit_bits_from_encoder(knowledge_vecs, t)

            y_tilde = self.process_bits_at_receiver(x, t, noise_ff, noise_fb)

            if t!=0:
                self.recvd_y_tilde = torch.cat((self.recvd_y_tilde,y_tilde.unsqueeze(-1)),axis=2)
                self.prev_xmit_signal = torch.cat((self.prev_xmit_signal, x.unsqueeze(-1)), axis=2)
            else:
                self.prev_xmit_signal = x.unsqueeze(-1)
                self.recvd_y_tilde = y_tilde.unsqueeze(-1)
            
            if t <= self.T-1: # don't need to update the feedback information after the last transmission.
                knowledge_vecs = self.make_knowledge_vecs(bitstreams, fb_info=self.recvd_y_tilde, prev_x=self.prev_xmit_signal)

        dec_out = self.decode_received_symbols(self.recvd_y)

        return dec_out

    #
    #
    def make_knowledge_vecs(self, b, fb_info=None, prev_x=None):
        if fb_info is None:
            fbi = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            px = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            q = torch.cat((px,fbi),axis=2)
        else:
            q = torch.cat((prev_x,fb_info),axis=2)
            q = F.pad(q, pad=(0, 2*(self.T - 1) - q.shape[-1]), value=-100)

        return torch.cat((b, q),axis=2)

    #
    # Do all the transmissions from the encoder side to the decoder side.
    def transmit_bits_from_encoder(self, k, t):
        x = self.embedding_encoder(k)
        x = self.pos_encoding_encoder(x)
        x = self.encoder(x, src_key_padding_mask = (k == -100)[:,:,0])
        x = self.enc_raw_output(x).squeeze(-1)
        x = self.normalize_transmit_signal_power(x, t)

        return x

    #
    # Process the received symbols at the decoder side. NOT THE DECODING STEP!!!
    def process_bits_at_receiver(self, x, t, noise_ff, noise_fb):
        self.transmit_power_tracking.append(torch.sum(torch.abs(x)**2,axis=1).detach().clone().cpu().numpy())
        y =  x + noise_ff[:,:,t]
        self.recvd_y[:,:,t] = y
        y_tilde = y + noise_fb[:,:,t]

        return y_tilde

    #
    # Actually decode all of the received symbols.
    def decode_received_symbols(self,y):
        y = self.embedding_decoder(y)
        y = self.pos_encoding_decoder(y)
        y = self.decoder(y)
        y = self.dec_raw_output(y)

        return y

    #
    # Make AWGN
    def generate_awgn(self,shape, noise_power):
        return sqrt(noise_power) * torch.randn(size=shape).to(self.device)

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
        x = torch.tanh(x)
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