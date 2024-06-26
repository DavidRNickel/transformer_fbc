import numpy as np
from math import sqrt, pi
from scipy.special import j0 #0th order Bessel function, generic softmax
import sys

import torch
from torch import nn
import torch.nn.functional as F

from attention_network import general_attention_network
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
        self.use_beliefs = conf.use_belief_network
        self.N = conf.N
        self.K = conf.K
        self.M = conf.M
        self.T = conf.T
        print(f'K: {self.K}, M: {self.M}, N: {self.N}, T: {self.T}')
        self.num_blocks = int(self.K // self.M)
        self.noise_pwr_ff = conf.noise_pwr_ff
        self.noise_pwr_fb = conf.noise_pwr_fb
        self.training = True
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        print('Making encoder...')
        # (self.embedding_encoder, self.pos_encoding_encoder, 
        #  self.encoder, self.enc_raw_output) = general_attention_network(dim_in=conf.knowledge_vec_len, dim_out=1, dim_embed=96, d_model=conf.d_model,
        #                                                   activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_xmit)
        (self.embedding_encoder, self.pos_encoding_encoder, 
         self.encoder, self.enc_raw_output) = general_attention_network(dim_in=conf.knowledge_vec_len, dim_out=1, dim_embed=96, d_model=conf.d_model, dim_embed_out=96,
                                                          activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_xmit)
        self.enc_neg_mask = torch.ones((self.T, self.num_blocks, conf.knowledge_vec_len)).to(self.device)
        for t in range(self.T):
            self.enc_neg_mask[t,:,self.M:self.M+t] *= -1
        self.emb_enc_lin_out = nn.Linear(192, conf.d_model)

        print('Making decoder...')
        # (self.embedding_decoder, self.pos_encoding_decoder, 
        #  self.decoder, self.dec_raw_output) = general_attention_network(dim_in=self.T, dim_out=2**self.M, dim_embed=96, d_model=conf.d_model, 
        #                                                   activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_recv)
        (self.embedding_decoder, self.pos_encoding_decoder, 
         self.decoder, self.dec_raw_output) = general_attention_network(dim_in=self.T, dim_out=2**self.M, dim_embed=96, d_model=conf.d_model, dim_embed_out=96,
                                                          activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_recv)
        self.emb_dec_lin_out = nn.Linear(192, conf.d_model)

        if self.use_beliefs:
            print('Making belief network...')
            # (self.embedding_belief, self.pos_encoding_belief, 
            #  self.belief_attn_nwk, self.belief_raw_output) = general_attention_network(dim_in=self.T-1, dim_out=2*self.M, dim_embed=96, d_model=conf.d_model, 
            #                                                                            activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_belief)
            (self.embedding_belief, self.pos_encoding_belief, 
             self.belief_attn_nwk, self.belief_raw_output) = general_attention_network(dim_in=self.T-1, dim_out=2*self.M, dim_embed=96, d_model=conf.d_model, dim_embed_out=96,
                                                                                       activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_belief)
            self.bel_neg_mask = torch.ones((self.T, self.num_blocks, self.T-1)).to(self.device)
            for t in range(self.T):
                self.bel_neg_mask[t,:,:t] *= -1
            self.emb_bel_lin_out = nn.Linear(192, conf.d_model)

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
        beliefs = None
        knowledge_vecs = self.make_knowledge_vecs(bitstreams.to(self.device))
        self.weight_power_normalized = torch.sqrt(self.weight_power**2 * (self.T) / (self.weight_power**2).sum())
        if noise_ff is None:
            noise_ff = sqrt(self.noise_pwr_ff) * torch.randn((self.batch_size, self.num_blocks, self.T), requires_grad=False).to(self.device)
            noise_fb = sqrt(self.noise_pwr_fb) * torch.randn((self.batch_size, self.num_blocks, self.T), requires_grad=False).to(self.device)
        else:
            noise_ff = noise_ff
            noise_fb = noise_fb

        # Initialize to this so we can pad it later on.
        # self.recvd_y = -100*torch.ones((self.batch_size, self.num_blocks, self.T)).to(self.device)
        self.recvd_y = torch.zeros((self.batch_size, self.num_blocks, self.T)).to(self.device)
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
            
            if t < self.T-1:
                if self.conf.use_belief_network:
                    beliefs = self.get_beliefs(self.make_belief_vecs(), t+1)

                # if t < self.T-1: # don't need to update the feedback information after the last transmission.
                knowledge_vecs = self.make_knowledge_vecs(bitstreams,
                                                        fb_info=self.recvd_y_tilde, 
                                                        prev_x=self.prev_xmit_signal,
                                                        beliefs=beliefs)

        dec_out = self.decode_received_symbols(self.recvd_y)

        return dec_out

    #
    #
    def make_knowledge_vecs(self, b, fb_info=None, prev_x=None, beliefs=None):
        if fb_info is not None:
            # px = F.pad(prev_x, pad=(0,self.T-1-prev_x.shape[-1]), value=-100)
            # fbi = F.pad(fb_info, pad=(0,self.T-1-fb_info.shape[-1]), value=-100)
            px = F.pad(prev_x, pad=(0,self.T-1-prev_x.shape[-1]), value=0)
            fbi = F.pad(fb_info, pad=(0,self.T-1-fb_info.shape[-1]), value=0)
            if self.use_beliefs == False:
                q = torch.cat((fbi, px),axis=2)
            else:
                q = torch.cat((fbi, px, beliefs),axis=2)

        else:
            # fbi = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            # px = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            fbi = torch.zeros(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            px = torch.zeros(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            if self.use_beliefs == False:
                q = torch.cat((fbi, px),axis=2)
            else:
                # bel = -100 * torch.ones(self.batch_size, self.num_blocks, 2*self.M).to(self.device)
                bel = torch.zeros(self.batch_size, self.num_blocks, 2*self.M).to(self.device)
                q = torch.cat((fbi, px, bel),axis=2)

        return torch.cat((b, q),axis=2)

    #
    # Do all the transmissions from the encoder side to the decoder side.
    def transmit_bits_from_encoder(self, k, t):
        x1 = self.embedding_encoder(k)
        x2 = self.embedding_encoder(self.enc_neg_mask[t]*k)
        x = self.emb_enc_lin_out(torch.cat((x1,x2),axis=2))
        # x = self.embedding_encoder(k)
        x = self.pos_encoding_encoder(x)
        x = self.encoder(x)
        x = self.enc_raw_output(x).squeeze(-1)
        x = self.tanh(x - x.mean())
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
    def decode_received_symbols(self, z):
        y1 = self.embedding_decoder(z)
        y2 = self.embedding_decoder(-1*z)
        y = self.emb_dec_lin_out(torch.cat((y1,y2),axis=2))
        # y = self.embedding_decoder(z)
        y = self.pos_encoding_decoder(y)
        y = self.decoder(y)
        y = self.dec_raw_output(y)

        return y

    #
    #
    def make_belief_vecs(self):
        # return F.pad(self.recvd_y_tilde, pad=(0, self.T-1 - self.recvd_y_tilde.shape[-1]), value=-100)
        return F.pad(self.recvd_y_tilde, pad=(0, self.T-1 - self.recvd_y_tilde.shape[-1]), value=0)

    #
    # Take in received information and make belief vectors.
    def get_beliefs(self, z, t):
        y1 = self.embedding_belief(z)
        y2 = self.embedding_belief(-1 * z)
        y = self.emb_bel_lin_out(torch.cat((y1,y2),axis=2))
        # y = self.embedding_belief(z)
        y = self.pos_encoding_belief(y)
        y = self.belief_attn_nwk(y)
        y = self.belief_raw_output(y)
        y = F.softmax(y.view(self.batch_size, -1, 2), dim=1)

        return y.view(self.batch_size, -1, 2*self.M)

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
        x = torch.stack([(b * (1<<torch.arange(b.shape[-1]-1,-1,-1).to(self.device))).sum(1) for b in bitstreams]).view(-1,1)

        return F.one_hot(x, num_classes=2**self.M).squeeze(1)

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
