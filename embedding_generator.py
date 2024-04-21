# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:29:51 2024

@author: dnick
"""
import numpy as np
from math import sqrt
import sys
import pickle as pkl

#
# embedding_generator.py
#
def embedding(z_vals, precision, bits, max_len):
    # Turn the boolean np array into a string of ones and zeros with no
    # square bracks on either side.
    bits = np.array2string(bits,separator='')[1:-1]
    z_prime_mags = np.abs((10**(precision)*z_vals)).astype(np.int16)
    bses = (z_vals < 0).astype(np.int8)
    embedded_reps = []
    for z, bs in zip(z_prime_mags, bses):
        b_z = f'{z:b}'.zfill(max_len)
        b_bin = f'{bits}' + f'{bs}' + b_z
        embedded_reps.append(int(b_bin,2))
    return embedded_reps

#
# Get the maximum possible length of the binary representations.
def get_max_len(s,precision):
    s *= 10**(precision)
    z_mag_max = f'{int(s):b}'
    return len(z_mag_max)

#
# Make an n-length vector of real AWGN with specified noise power.
def generate_awgn(n, noise_power):
    return np.random.normal(0,sqrt(noise_power),n)

#
# Make all of the inputs into the Encoder for a given code.
def make_embedded_inputs(N, bitstream, fwd_noise_pwr, fbk_noise_pwr, max_len, precision, s):
    embedded_data = -1*np.ones((N,N))
    fwd_noise = np.clip(generate_awgn(N, fwd_noise_pwr), -s, s)
    fbk_noise = np.clip(generate_awgn(N, fbk_noise_pwr), -s, s)
    fbk_noise[0] = 0
    noise_sum = fwd_noise + fbk_noise
    eds = embedding(noise_sum, precision, bitstream, max_len)
    for n in range(N):
        embedded_data[n,:n+1] = eds[:n+1]

    # need to add in the fwd noise in the Xformer model
    return np.array(embedded_data), fwd_noise, fbk_noise


if __name__=='__main__':
    # TODO: need to make embedding() do the entire bitstream for a single noise_sum value
    # make it: (bits, sign, noise)
    s = 9.99
    precision = 2
    max_len = get_max_len(9.99, precision)
    N = 18
    K = 6
    bitstreams = []
    fwd_noises = []
    embedded_data = []
    for _ in range(100):
        bits = np.random.randint(0,2,K)
        data,fwd_noise,n = make_embedded_inputs(N, bits, 1, .1, max_len, precision, s)
        bitstreams.append(bits)
        fwd_noises.append(fwd_noise)
        embedded_data.append(data)
    
    outdata = {'bitstreams' : bitstreams,
               'fwd_noises' : fwd_noises,
               'embedded_data' : embedded_data}
    with open('testoutput.pkl','wb') as f:
        pkl.dump(outdata, f)
