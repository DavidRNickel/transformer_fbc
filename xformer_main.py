import numpy as np
from math import sqrt

import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from config_class import Config

ONE_OVER_SQRT_TWO = 1/np.sqrt(2)

#
# Make AWGN
def generate_awgn(shape, noise_power):
    noise = np.random.normal(0,ONE_OVER_SQRT_TWO,shape) + 1j*np.random.normal(0,ONE_OVER_SQRT_TWO,shape)
    
    return sqrt(noise_power) * noise


#
# Make Rayleigh fading channels
def generate_channel_gains_rayleigh(self, shape, prev_h=None, is_pu_su=False):
        if self.is_symmetric_gains and not is_pu_su:
            temp = np.tril(self.generate_AWGN(shape=shape, noise_power=1))
            e = np.empty(shape=shape,dtype=np.complex64)
            if len(shape) == 3:
                for k in range(self.K):
                    e[k] = temp[k] + temp[k].T - np.diag(temp[k].diagonal())
            else:
                e = temp + temp.T - np.diag(temp.diagonal())

            # multiply by random phase
            e *= np.exp(1j * self.rng.uniform(0, 2*pi, size=shape))

        else:
            e = self.generate_AWGN(shape=shape, noise_power=1)

        if prev_h is not None:
            # h(t) = rho*h(t-1) + sqrt(1-rho^2)*e(t)
            return self.RHO*prev_h + self.SQRT_ONE_MIN_RHO_2*e

        else:
            # h(t) = e(t)
            return e

if __name__=='__main__':
    conf = Config()

    enc_layer = TransformerEncoderLayer(d_model=conf.d_model,
                                         nhead=conf.n_heads,
                                         norm_first=True,
                                         dropout=conf.drop_prob,
                                         activation='gelu')
    encoder = TransformerEncoder(enc_layer, num_layers=2)

    embedding_layer = nn.Linear(conf.knowledge_vec_len,conf.d_model)

    bitstreams = np.random.randint(0,2,(conf.batch_size, 1, conf.K))
    H = generate_channel_gains_rayleigh(shape=(conf.batch_size, 1, conf.num_xmit_chans))

    # print(encoder(embedding_layer(src)).shape)