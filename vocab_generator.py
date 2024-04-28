# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:29:51 2024

@author: dnick
"""
import numpy as np
from math import sqrt
import sys
import pickle as pkl

class VocabGenerator():
    def __init__(self, 
                 precision_enc, precision_rec,
                 s_enc, s_rec,
                 K, N):
        self.precision_enc = precision_enc
        self.precision_rec = precision_rec

        # These are the biggest allowed mangitudes for the data.
        # Ex: s_rec = 5 and receive [-10,-3,3,10]-->[-5,-3,3,5]
        self.s_enc = s_enc
        self.s_rec = s_rec
        self.max_len = self.get_max_len_enc()
        
        self.K = K
        self.N = N

    #
    # 
    def make_encoder_words(self, z_vals, bits):
        # Turn the boolean np array into a string of ones and zeros with no
        # square bracks on either side.
        bits = np.array2string(bits,separator='')[1:-1]
        z_prime_mags = np.abs((10**self.precision_enc * z_vals)).astype(np.int16)
        bses = (z_vals < 0).astype(np.int8)
        translated_reps = []
        for z, bs in zip(z_prime_mags, bses):
            b_z = f'{z:b}'.zfill(self.max_len)
            b_bin = f'{bits}' + f'{bs}' + b_z
            translated_reps.append(int(b_bin,2))
        return translated_reps 

    #
    #
    def make_received_vocab(self, z_vals):
        z_max = self.s_rec * 10**self.precision_rec
        z_mags = (10**self.precision_rec * z_vals + z_max).astype(np.int16)
        return np.clip(z_mags, 0, 2*self.s_rec) 

    #
    # Get the maximum possible length of the binary representations.
    def get_max_len_enc(self):
        z_mag_max = f'{int(self.s_enc * 10**self.precision_enc):b}'
        return len(z_mag_max)

    #
    # Make an n-length vector of real AWGN with specified noise power.
    def generate_awgn(self, n, noise_power):
        return np.random.normal(0,sqrt(noise_power),n)

    #
    # Make all of the inputs into the Encoder for a given code.
    def make_encoder_inputs(self, N, bitstream, fwd_noise_pwr, fbk_noise_pwr, max_len, precision, s):
        output_words = -1*np.ones((N,N))
        fwd_noise = np.clip(self.generate_awgn(N, fwd_noise_pwr), -self.s_enc, self.s_enc)
        fbk_noise = np.clip(self.generate_awgn(N, fbk_noise_pwr), -self.s_enc, self.s_enc)
        fbk_noise[0] = 0
        noise_sum = fwd_noise + fbk_noise
        words = self.make_encoder_words(noise_sum, bitstream)
        for n in range(N):
            output_words[n,:n+1] = words[:n+1]

        # need to add in the fwd noise in the Xformer model
        return np.array(output_words), fwd_noise, fbk_noise

    #
    # Used for setting the size of the vocabulary in the encoder.
    def get_max_input_encoder(self):
        return int(self.make_encoder_words(np.array([-self.s_enc]), self.precision_enc, np.array([1 for _ in range(self.K)]).astype(np.int16), self.max_len)[0])

    #
    # Used for setting the size of the vocabulary in the receiver.
    def get_max_input_receiver(self):
        return int(2 * self.s_rec * 10**self.precision_rec)

    #
    # Make train, test, eval data by writing it to a file.
    def generate_encoder_dataset(self, num_bitstreams, outfile=None, fwd_noise_pwr=1, fbk_noise_pwr=.1):
        bitstreams = []
        fwd_noises = []
        encoder_inputs = []
        for _ in range(num_bitstreams):
            bits = np.random.randint(0,2,self.K)
            data, fwd_noise, _ = self.make_encoder_inputs(self.N, bits, fwd_noise_pwr, fbk_noise_pwr, self.max_len, self.precision_enc, self.s_enc)
            bitstreams.append(bits)
            fwd_noises.append(fwd_noise)
            encoder_inputs.append(data)
        
        outdata = {'bitstreams' : bitstreams,
                   'fwd_noises' : fwd_noises, 
                   'encoder_inputs' : np.array(encoder_inputs, dtype=np.int64)}

        if outfile is not None:
            with open(outfile,'wb') as f:
                pkl.dump(outdata, f)

        return outdata


if __name__=='__main__':
    # TODO: need to make embedding() do the entire bitstream for a single noise_sum value
    # make it: (bits, sign, noise)
    vocab_gen = VocabGenerator(precision_enc=2, precision_rec=4,
                               s_enc=2.55, s_rec=4.5,
                               K=6, N=18)

    # Convert the bitstreams and noises into a finite vocabulary which the
    # encoder is capable of handling.
    outdata = vocab_gen.generate_encoder_dataset(num_bitstreams=100)
    print(outdata['bitstreams'][0][:10])
    print(outdata['fwd_noises'][0][:10])
    print(outdata['encoder_inputs'][0])

    # Translate the received vector (y) into a finite vocabulary.
    # This information is then passed into an embedding layer which
    # the decoder is then capable of processing.
    print(vocab_gen.get_max_input_receiver())
    