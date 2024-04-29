import numpy as np
import matplotlib.pyplot as plt
import random
import decimal
import scipy.special
import math
from copy import copy



def add_awgn(signal, ebno_db):
    out_data = np.zeros(len(signal), dtype=np.float32)
    ebno = 10.0**(ebno_db/10.0)
    noise_pow = 1/np.sqrt(2*ebno)
    noise = noise_pow * np.random.randn(len(signal))
    out_data = signal + noise
        
    return out_data


if __name__=='__main__':
    K = 2304
    NUM_DECODER_ITERS = 5
    RHO = 1
    
    N,s,r = 4,.1,5
    g = np.ones(N).reshape(-1,1)/(np.sqrt(N))
    gamma = .3562
    b = .7247
    q = np.sqrt((1-b**2)/(1-b**(2*N)))*np.array([1,b,b**2,b**3]).reshape(-1,1)
    F_coef = -1*(1-b**2)/((1+s)*b)
    F = F_coef * np.array([[0,    0, 0, 0],
                           [1,    0, 0, 0],
                           [b,    1, 0, 0],
                           [b**2, b, 1, 0]])
    
    
    
    """
    # set up the LDPC encoder / decoder
    modem = mod.PSKModem(m=4)

    alist_file = f"{LDPC_PATH}/alist/wimax_{K}_0_5.alist"
    encoder = ldpc_encoder.ldpc_encoder(alist_file, 5, 7, False)
    decoder = ldpc_decoder.ldpc_decoder(alist_file)

    data = np.random.randint(0,2, K//2, dtype=np.uint8)
    encoded_data = encoder.encode_data(data)
    print("Num Errors: %d\n" % decoder.compute_syndrome(encoded_data))
    
    decoded_data = decoder.ldpc_tdmp(encoded_data, NUM_DECODER_ITERS)
    print((decoded_data[0:K//2] == data).all())

    # Generate the data
    rounds = 20
    data = np.random.randint(0,2,encoder.N//2)
    encoded_data = encoder.encode_data(data)
    print(encoded_data[:5])
    modulated_data = -2.0 * encoded_data + 1.0
    print(modulated_data[:5])

    ebno_dBs = np.linspace(-20,15,20)
    ber_results = []
    for i in ebno_dBs:
        print(i)
        errors = 0
        for k in range(rounds):
            # apply noise
            received_data = add_awgn(modulated_data, i)

            decoded_data = decoder.ldpc_tdmp(received_data, NUM_DECODER_ITERS)
            errors += (decoded_data[0:K//2] != data).sum()
            
        ber_results.append(1.0 * errors / (rounds*(decoder.N/2)))

    # Plot the modulated data in the complex plane
    plt.figure()
    plt.semilogy(ebno_dBs, ber_results, 'r')
    plt.axis([-20, 15, 1e-5, 1])
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('EbNo(dB)')
    plt.ylabel('BER')
    plt.title('BPSK Bit Error Rate Curve')
    """