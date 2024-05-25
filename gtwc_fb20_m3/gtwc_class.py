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
# fd = 10
# T = 100E-3
# RHO = j0(2*pi*fd*T)
# SQRT_ONE_MIN_RHO_2 = sqrt(1 - RHO**2)


class GTWC(nn.Module):
    def __init__(self, conf):
        super(GTWC, self).__init__()

        # Shared parameters across both users.
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

        (self.emb_enc_1, self.pos_enc_enc_1, 
         self.enc_1, self.enc_raw_out_1) = general_attention_network(dim_in=conf.knowledge_vec_len, dim_out=1, dim_embed=96, d_model=conf.d_model,
                                                                     activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_xmit)

        (self.emb_dec_1, self.pos_enc_dec_1, 
         self.dec_1, self.dec_raw_out_1) = general_attention_network(dim_in=self.T+self.M, dim_out=2**self.M, dim_embed=96, d_model=conf.d_model, 
                                                                     activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_recv)

        (self.emb_enc_2, self.pos_enc_enc_2, 
         self.enc_2, self.enc_raw_out_2) = general_attention_network(dim_in=conf.knowledge_vec_len, dim_out=1, dim_embed=96, d_model=conf.d_model,
                                                                     activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_xmit)

        (self.emb_dec_2, self.pos_enc_dec_2, 
         self.dec_2, self.dec_raw_out_2) = general_attention_network(dim_in=self.T+self.M, dim_out=2**self.M, dim_embed=96, d_model=conf.d_model, 
                                                                     activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_recv)

        if self.use_beliefs:
            (self.emb_bel_1, self.pos_enc_bel_1, 
             self.bel_nwk_1, self.bel_raw_out_1) = general_attention_network(dim_in=self.T-1, dim_out=2*self.M, dim_embed=96, d_model=conf.d_model, 
                                                                             activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_belief)

            (self.emb_bel_2, self.pos_enc_bel_2, 
             self.bel_nwk_2, self.bel_raw_out_2) = general_attention_network(dim_in=self.T-1, dim_out=2*self.M, dim_embed=96, d_model=conf.d_model, 
                                                                             activation=self.relu, max_len=self.num_blocks, num_layers=conf.num_layers_belief)

        # Power weighting-related parameters.
        self.wgt_pwr_1 = torch.nn.Parameter(torch.Tensor(self.T), requires_grad=True)
        self.wgt_pwr_1.data.uniform_(1., 1.)
        self.wgt_pwr_normed_1 = torch.sqrt(self.wgt_pwr_1**2 * (self.T)/torch.sum(self.wgt_pwr_1**2))
        self.xmit_pwr_track_1 = []

        self.wgt_pwr_2 = torch.nn.Parameter(torch.Tensor(self.T), requires_grad=True)
        self.wgt_pwr_2.data.uniform_(1., 1.)
        self.wgt_pwr_normed_2 = torch.sqrt(self.wgt_pwr_2**2 * (self.T)/torch.sum(self.wgt_pwr_2**2))
        self.xmit_pwr_track_2 = []

        # Parameters for normalizing mean and variance of transmit signals.
        self.mean_batch_1 = torch.zeros(self.T)
        self.std_batch_1 = torch.ones(self.T)
        self.mean_saved_1 = torch.zeros(self.T)

        self.mean_batch_2 = torch.zeros(self.T)
        self.std_batch_2 = torch.ones(self.T)
        self.mean_saved_2 = torch.zeros(self.T)
        self.normalization_with_saved_data = False # True: inference w/ saved mean, var; False: calculate mean, var

    #
    #
    def forward(self, bitstreams_1, bitstreams_2, noise_ff=None, noise_fb=None):
        know_vecs_1, know_vecs_2 = self.make_knowledge_vecs(b=(bitstreams_1, bitstreams_2))
        beliefs_1 = None
        beliefs_2 = None

        self.wgt_pwr_normed_1 = torch.sqrt(self.wgt_pwr_1**2 * (self.T) / (self.wgt_pwr_1**2).sum())
        self.wgt_pwr_normed_2 = torch.sqrt(self.wgt_pwr_2**2 * (self.T) / (self.wgt_pwr_2**2).sum())

        if noise_ff is None:
            noise_ff = sqrt(self.noise_pwr_ff) * torch.randn((self.batch_size, self.num_blocks, self.T)).to(self.device)
            noise_fb = sqrt(self.noise_pwr_fb) * torch.randn((self.batch_size, self.num_blocks, self.T)).to(self.device)

        self.recvd_y_1 = None # dummy initializations; populated in process_bits_rx()
        self.recvd_y_2 = None

        self.xmit_pwr_track_1 = []
        self.xmit_pwr_track_2 = []

        for t in range(self.T):
            # Transmit side
            x1, x2 = self.transmit_bits_from_encoder(know_vecs_1, know_vecs_2, t)

            self.process_bits_at_receiver(x1, x2, t, noise_ff, noise_fb)

            if t != 0:
                self.prev_xmit_signal_1 = torch.cat((self.prev_xmit_signal_1, x1.unsqueeze(-1)), axis=2)
                self.prev_xmit_signal_2 = torch.cat((self.prev_xmit_signal_2, x2.unsqueeze(-1)), axis=2)
            else:
                self.prev_xmit_signal_1 = x1.unsqueeze(-1)
                self.prev_xmit_signal_2 = x2.unsqueeze(-1)
            
            if self.use_beliefs:
                bv1, bv2 = self.make_belief_vecs()
                beliefs_1, beliefs_2 = self.get_beliefs(bv1, bv2)

            know_vecs_1, know_vecs_2 = self.make_knowledge_vecs(b=(bitstreams_1, bitstreams_2), 
                                                                fb_info=(self.recvd_y_1, self.recvd_y_2), 
                                                                prev_x=(self.prev_xmit_signal_1, self.prev_xmit_signal_2), 
                                                                beliefs=(beliefs_1, beliefs_2) if self.use_beliefs else None)

        dec_out_1, dec_out_2 = self.decode_received_symbols(torch.cat((self.recvd_y_2, bitstreams_2), axis=2),
                                                            torch.cat((self.recvd_y_1, bitstreams_1), axis=2))

        return dec_out_1, dec_out_2

    #
    #
    def make_knowledge_vecs(self, b, fb_info=None, prev_x=None, beliefs=None):
        if fb_info is None:
            fbi_1 = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            fbi_2 = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            px_1 = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            px_2 = -100 * torch.ones(self.batch_size, self.num_blocks, self.T - 1).to(self.device)
            if self.use_beliefs == False:
                q_1 = torch.cat((px_1, fbi_1),axis=2)
                q_2 = torch.cat((px_2, fbi_2),axis=2)
            else:
                bel_1 = -100 * torch.ones(self.batch_size, self.num_blocks, 2*self.M).to(self.device)
                bel_2 = -100 * torch.ones(self.batch_size, self.num_blocks, 2*self.M).to(self.device)
                q_1 = torch.cat((px_1, fbi_1, bel_1),axis=2)
                q_2 = torch.cat((px_2, fbi_2, bel_2),axis=2)
        else:
            fbi_1, fbi_2 = fb_info
            px_1, px_2 = prev_x
            px_1 = F.pad(px_1, pad=(0,self.T-1-px_1.shape[-1]), value=-100)
            px_2 = F.pad(px_2, pad=(0,self.T-1-px_2.shape[-1]), value=-100)
            fbi_1 = F.pad(fbi_1, pad=(0,self.T-1-fbi_1.shape[-1]), value=-100)
            fbi_2 = F.pad(fbi_2, pad=(0,self.T-1-fbi_2.shape[-1]), value=-100)
            if self.use_beliefs == False:
                q_1 = torch.cat((px_1, fbi_1),axis=2)
                q_2 = torch.cat((px_2, fbi_2),axis=2)
            else:
                bel_1, bel_2 = beliefs
                bel_1 = F.pad(bel_1, pad=(0,2*self.M-bel_1.shape[-1]), value=-100)
                bel_2 = F.pad(bel_2, pad=(0,2*self.M-bel_2.shape[-1]), value=-100)
                q_1 = torch.cat((px_1, fbi_1, bel_1),axis=2)
                q_2 = torch.cat((px_2, fbi_2, bel_2),axis=2)

        b_1, b_2 = b

        return torch.cat((b_1, q_1),axis=2), torch.cat((b_2, q_2),axis=2)
    
    #
    #
    def make_belief_vecs(self):
        bv1 = F.pad(self.recvd_y_1, pad=(0, self.T-1 - self.recvd_y_1.shape[-1]), value=-100)
        bv2 = F.pad(self.recvd_y_2, pad=(0, self.T-1 - self.recvd_y_2.shape[-1]), value=-100)

        return bv1, bv2

    #
    # Do all the transmissions from the encoder side to the decoder side.
    def transmit_bits_from_encoder(self, k1, k2, t):
        x1 = self.emb_enc_1(k1)
        x1 = self.pos_enc_enc_1(x1)
        x1 = self.enc_1(x1, src_key_padding_mask = (k1 == -100)[:,:,0])
        x1 = self.enc_raw_out_1(x1).squeeze(-1)
        x1 = self.tanh(x1 - x1.mean())

        x2 = self.emb_enc_2(k2)
        x2 = self.pos_enc_enc_2(x2)
        x2 = self.enc_2(x2, src_key_padding_mask = (k2 == -100)[:,:,0])
        x2 = self.enc_raw_out_2(x2).squeeze(-1)
        x2 = self.tanh(x2 - x2.mean())

        return self.normalize_transmit_signal_power(x1, x2, t)

    #
    # Process the received symbols at the decoder side. NOT THE DECODING STEP!!!
    def process_bits_at_receiver(self, x1, x2, t, noise_ff, noise_fb):
        self.xmit_pwr_track_1.append(torch.sum(torch.abs(x1)**2,axis=1).detach().clone().cpu().numpy())
        self.xmit_pwr_track_2.append(torch.sum(torch.abs(x2)**2,axis=1).detach().clone().cpu().numpy())

        y2 =  x1 + noise_ff[:,:,t]
        y1 =  x2 + noise_fb[:,:,t]

        if t != 0:
            self.recvd_y_1 = torch.cat((self.recvd_y_1, y1.unsqueeze(-1)), axis=2)
            self.recvd_y_2 = torch.cat((self.recvd_y_2, y2.unsqueeze(-1)), axis=2)
        else:
            self.recvd_y_1 = y1.unsqueeze(-1)
            self.recvd_y_2 = y2.unsqueeze(-1)
        
        return

    #
    # Actually decode all of the received symbols.
    def decode_received_symbols(self, y1, y2):
        y1 = self.emb_dec_1(y1)
        y1 = self.pos_enc_dec_1(y1)
        y1 = self.dec_1(y1)
        y1 = self.dec_raw_out_1(y1)

        y2 = self.emb_dec_2(y2)
        y2 = self.pos_enc_dec_2(y2)
        y2 = self.dec_2(y2)
        y2 = self.dec_raw_out_2(y2)

        return y1, y2

    #
    # Take in received information and make belief vectors.
    def get_beliefs(self, y1, y2):
        y1 = self.emb_bel_1(y1)
        y1 = self.pos_enc_bel_1(y1)
        y1 = self.bel_nwk_1(y1)
        y1 = self.bel_raw_out_1(y1)
        y1 = F.softmax(y1.view(self.batch_size, -1, 2), dim=1)

        y2 = self.emb_bel_2(y2)
        y2 = self.pos_enc_bel_2(y2)
        y2 = self.bel_nwk_2(y2)
        y2 = self.bel_raw_out_2(y2)
        y2 = F.softmax(y2.view(self.batch_size, -1, 2), dim=1)

        return y1.view(self.batch_size, -1, 2*self.M), y2.view(self.batch_size, -1, 2*self.M)

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
    def normalize_transmit_signal_power(self, x1, x2, t):
        x1 = self.normalization(x1, t, 1)
        x2 = self.normalization(x2, t, 2)

        return self.wgt_pwr_normed_1[t] * x1, self.wgt_pwr_normed_2[t] * x2

    #
    # Normalize the batch.
    def normalization(self, inputs, t_idx, uid):
        if uid==1:
            mean_batch_1 = torch.mean(inputs)
            std_batch_1 = torch.std(inputs)
            if self.training == True:
                outputs = (inputs - mean_batch_1) / std_batch_1
            else:
                if self.normalization_with_saved_data:
                    outputs = (inputs - self.mean_saved_1[t_idx]) / self.std_saved_1[t_idx]
                else:
                    self.mean_batch_1[t_idx] = mean_batch_1
                    self.std_batch_1[t_idx] = std_batch_1
                    outputs = (inputs - mean_batch_1) / std_batch_1

        else:
            mean_batch_2 = torch.mean(inputs)
            std_batch_2 = torch.std(inputs)
            if self.training == True:
                outputs = (inputs - mean_batch_2) / std_batch_2
            else:
                if self.normalization_with_saved_data:
                    outputs = (inputs - self.mean_saved_2[t_idx]) / self.std_saved_2[t_idx]
                else:
                    self.mean_batch_2[t_idx] = mean_batch_2
                    self.std_batch_2[t_idx] = std_batch_2
                    outputs = (inputs - mean_batch_2) / std_batch_2

        return outputs
