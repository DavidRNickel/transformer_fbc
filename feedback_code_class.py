import numpy as np
from math import sqrt, pi
from scipy.special import j0 #0th order Bessel function, generic softmax
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from config_class import Config
from positional_encoding_class import PositionalEncoding

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
        self.M_t = conf.num_xmit_chans
        self.noise_pwr_ff = conf.noise_pwr_ff
        self.noise_pwr_fb = conf.noise_pwr_fb
        self.training = True
        self.activation = nn.GELU()

        # Set up the transmit side encoder.
        self.enc_layer = TransformerEncoderLayer(d_model=conf.d_model, 
                                                 nhead=conf.n_heads, 
                                                 norm_first=True, 
                                                 dropout=conf.dropout, 
                                                 dim_feedforward=conf.scaling_factor*conf.d_model,
                                                 activation=self.activation)
        self.encoder = TransformerEncoder(self.enc_layer, num_layers=conf.num_layers_xmit)
        self.embedding_encoder = nn.Sequential(nn.Linear(conf.knowledge_vec_len, conf.d_model), 
                                               self.activation, 
                                               nn.Linear(conf.d_model,conf.d_model), 
                                               self.activation)
        self.pos_encoding_encoder = PositionalEncoding(conf.d_model, dropout=conf.dropout)
        self.enc_raw_output = nn.Linear(self.d_model, 2*self.M_t)

        # Set up the receive side decoder.
        self.dec_layer = TransformerEncoderLayer(d_model=conf.d_model,
                                                 nhead=conf.n_heads,
                                                 norm_first=True,
                                                 dropout=conf.dropout,
                                                 activation=self.activation)
        self.decoder = TransformerEncoder(self.dec_layer, num_layers=conf.num_layers_recv)
        self.embedding_decoder = nn.Sequential(nn.Linear(2*self.N, conf.d_model), 
                                               self.activation, 
                                               nn.Linear(conf.d_model,conf.d_model), 
                                               self.activation)
        self.pos_encoding_decoder = PositionalEncoding(conf.d_model, dropout=conf.dropout)
        self.dec_raw_output = nn.Linear(self.d_model, 2**self.K)

        # Power weighting-related parameters.
        self.weight_power = torch.nn.Parameter(torch.Tensor(self.N), requires_grad=True )
        self.weight_power.data.uniform_(1., 1.)
        self.weight_power_normalized = torch.sqrt(self.weight_power**2 * (self.N)/torch.sum(self.weight_power**2))

        # Parameters for normalizing mean and variance of 
        self.mean_batch = torch.zeros(self.N)
        self.std_batch = torch.ones(self.N)
        self.mean_saved = torch.zeros(self.N)
        self.normalization_with_saved_data = False # True: inference w/ saved mean, var; False: calculate mean, var

    #
    # forward() calls both encoder and decoder
    def forward(self, knowledge_vecs, H_real, H_imag):
        self.weight_power_normalized = torch.sqrt(self.weight_power**2 * (self.N) / (self.weight_power**2).sum())
        noise_ff = sqrt(self.noise_pwr_ff) * torch.randn((self.batch_size, self.N)).to(self.device)
        noise_fb = sqrt(self.noise_pwr_fb) * torch.randn((self.batch_size, self.N)).to(self.device)
        self.recvd_y = torch.empty((self.batch_size, self.N), dtype=torch.cfloat).to(self.device)

        # Transmit side
        for t in range(self.N):
            x = self.transmit_bits_from_encoder(knowledge_vecs, t)

            # Receive side. dec_out~batch_size x 2^K; y_tilde~batch_size x 1
            y_tilde = self.process_bits_at_receiver(x, t, H_real, H_imag, noise_ff, noise_fb)
            
            # Update the knowledge vectors
            H_real, H_imag = self.generate_split_channel_gains_rayleigh(shape=(self.batch_size, self.M_t))
            H_prime = torch.cat((H_real,H_imag),axis=1)

            if t < self.N-1: # don't need to update the feedback information after the last transmission.
                knowledge_vecs[:,0,self.K : self.K + 2*self.M_t] = H_prime
                knowledge_vecs[:,0,-2*self.N + 2*t: -2*self.N + 2*t + 2] = torch.view_as_real(y_tilde)

        dec_out = self.decode_received_symbols(self.recvd_y)
        return dec_out

    #
    # Do all the transmissions from the encoder side to the decoder side.
    def transmit_bits_from_encoder(self, x, t):
        x = self.embedding_encoder(x)
        x = self.pos_encoding_encoder(x)
        x = self.encoder(x)
        x = self.enc_raw_output(x).squeeze(1)
        x = self.normalize_transmit_signal_power(x, t)

        return x

    #
    # Process the received symbols at the decoder side. NOT THE DECODING STEP!!!
    def process_bits_at_receiver(self, x, t, H_real, H_imag, noise_ff, noise_fb):
        x = x[:,::2] + 1j*x[:,1::2]
        H = H_real + 1j*H_imag
        y =  H * x + noise_ff[:,t].view(-1,1)
        y = y.sum(1)
        y_tilde = y + noise_fb[:,t]
        self.recvd_y[:,t] = y

        return y_tilde

    #
    # Actually decode all of the received symbols.
    def decode_received_symbols(self,y):
        y = torch.cat((y.real, y.imag),axis=1).unsqueeze(1)
        y = self.embedding_decoder(y)
        y = self.pos_encoding_decoder(y)
        y = self.decoder(y)

        return self.dec_raw_output(y.squeeze(1))

    #
    # Make AWGN
    def generate_awgn(self,shape, noise_power):
        return sqrt(noise_power) * torch.randn(size=shape,dtype=torch.cfloat).to(self.device)

    #
    # Make Rayleigh fading channels
    def generate_split_channel_gains_rayleigh(self,shape):
            chan = self.generate_awgn(shape=shape, noise_power=1)
            return chan.real, chan.imag

    #
    # The following methods are from https://anonymous.4open.science/r/RCode1/main_RobustFeedbackCoding.ipynb
    # which is the code for the paper "Robust Non-Linear Feedback Coding via Power-Constrained Deep Learning".
    #

    #
    # Handle the power weighting on the transmit bits.
    def normalize_transmit_signal_power(self, x, t):
        x = torch.tanh(x)
        x = self.normalization(x, t)

        return self.weight_power_normalized[t] * (1/sqrt(2*self.M_t)) * x 

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
    # Take the input bitstreams and map them to their one-hot representation.
    def bits_to_one_hot(self, bitstreams):
        # This is a torch adaptation of https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
        # It maps binary representations to their one-hot values by first converting the rows into 
        # the base-10 representation of the binary.
        x = (bitstreams * (1<<torch.arange(bitstreams.shape[-1]-1,-1,-1).to(self.device))).sum(1)

        return F.one_hot(x, num_classes=2**self.K)

    #
    # Map the onehot representations into their binary representations.
    def one_hot_to_bits(self, onehots):
        x = torch.argmax(onehots,dim=1)
        # Adapted from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        bin_representations = (((x[:,None] & (1 << torch.arange(self.K).to(self.device).flip(0)))) > 0).int()

        return bin_representations