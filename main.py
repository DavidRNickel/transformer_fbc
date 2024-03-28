# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:59:09 2024

@author: dnick
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

import sys
LDPC_PATH = './wimax_ldpc_lib/'
sys.path.append(f'{LDPC_PATH}/python_ldpc')

import ldpc_encoder
import ldpc_decoder

from chance_love_utils import calc_gamma_opt, calc_F, calc_q #,calc_beta_opt

def add_awgn(signal, ebno_db):
    out_data = np.zeros(len(signal), dtype=np.float32)
    ebno = 10.0**(ebno_db/10.0)
    noise_pow = 1/np.sqrt(2*ebno)
    noise = noise_pow * np.random.randn(len(signal))
    out_data = signal + noise
    print(f'signal: {signal[:10]}')
    print(f'out: {out_data[:10]}\n')
        
    return out_data

#
#
def generate_awgn_from_db(n, ebno_db):
    ebno = 10**(ebno_db/10)
    noise_power = 1/sqrt(2*ebno)
    return noise_power * np.random.randn(n).reshape(-1,1)

#
#
def generate_awgn(n, noise_power):
    return np.random.normal(0,sqrt(noise_power),n).reshape(-1,1)

#
#
def inner_code(symbols, s_2, n, ebno_db, gg):
    rho = 10**(ebno_db/10)
    gamma_opt = calc_gamma_opt(s_2, rho, n)
    beta_opt = np.sqrt((n - 1) / (n+(1+s_2)*n*gamma_opt*rho)) #calc_beta_opt(s_2, n, gamma_opt, rho)
    q = calc_q(beta_opt,n)
    F = calc_F(s_2, beta_opt, n)
    outdata = []
    for sym in sqrt(rho)*symbols:
        fwd_noise = generate_awgn(n, 1)
        fbk_noise = generate_awgn(n, s_2)   
        y = F@(fwd_noise+fbk_noise) + sym*gg + fwd_noise
        outdata.append((q.T @ y).item())

    # print(f'rho: {rho}')
    # print(f'gamma: {gamma_opt}')
    # print(f'beta: {beta_opt}')
    # print(f'q: {q.T}')
    # print(f'F: {F}\n')
    
    return outdata


if __name__=='__main__':
    N = 10 # length of inner code
    g = np.ones(N).reshape(-1,1)/sqrt(N)
    sigma_2 = .1
    num_constellation_symbols = 2


    K = 576
    NUM_DECODER_ITERS = 10
    alist_file = f'{LDPC_PATH}/alist/wimax_{K}_0_5.alist'
    encoder = ldpc_encoder.ldpc_encoder(alist_file, 5, 7, False)
    decoder = ldpc_decoder.ldpc_decoder(alist_file)
    
    rounds = 15
    ebdb_low, ebdb_high = -20,15
    ebno_dBs = np.linspace(ebdb_low, ebdb_high,10)
    ebno_linears = 10**(ebno_dBs/10)
    ber_results = []
    plt.figure()
    for eb_dB, eb_lin in zip(ebno_dBs, ebno_linears):
        print(round(eb_dB,5))
        errors = 0
        for k in range(rounds):
            data = np.random.randint(0,2,encoder.N//2)
            encoded_data = encoder.encode_data(data)
            modulated_data = -2.0 * encoded_data + 1.0 # cheap BPSK
            
            # received_data = add_awgn(modulated_data,eb_dB) 
            received_data = inner_code(modulated_data, sigma_2, N, eb_dB, g)

            decoded_data = decoder.ldpc_tdmp(received_data, NUM_DECODER_ITERS)
            errors += (decoded_data[0:K//2] != data).sum()
            
        ber_results.append(errors / (rounds*(decoder.N/2)))

    # Plot the modulated data in the complex plane
    plt.semilogy(ebno_dBs, ber_results, 'r')
    plt.axis([ebdb_low, ebdb_high+.1, 1e-5, 1])
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.title('BPSK Bit Error Rate Curve')
    plt.show()