import numpy as np
import torch
class params():
    def __init__(self):

        # Encoder
        self.encoder_act_func = 'tanh'
        self.encoder_N_layers: int = 2    # number of RNN layers at encoder
        self.encoder_N_neurons: int = 50  # number of neurons at each RNN
        
        # Decoder
        self.decoder_N_layers: int = 2    # number of RNN layers at decoder
        self.decoder_N_neurons: int = 50  # number of neurons at each RNN
        self.decoder_bidirection = True   # True: bi-directional decoding, False: uni-directional decoding
        self.attention_type: int = 5      # choose the attention type among five options
        # 1. Only the last timestep (N-th)
        # 2. Merge the last outputs of forward/backward RNN
        # 3. Sum over all timesteps
        # 4. Attention mechanism with N weights (same weight for forward/backward)
        # 5. Attention mechanism with 2N weights (separate weights for forward/backward)
        
        # Setup
        self.N_bits: int = 3                # number of bits
        self.N_channel_use = 9             # number of channel uses
        self.input_type = 'bit_vector'      # choose 'bit_vector' or 'one_hot_vector'
        self.output_type = 'one_hot_vector' # choose 'bit_vector' or 'one_hot_vector'

        # Learning parameters
        self.batch_size = int(2.5e3) 
        self.learning_rate = 0.01 
        self.use_cuda = True


def error_rate_bitvector(b_est, b):
    b = np.round(b)          # (batch,K)
    b_est = np.round(b_est)  # (batch,K)

    error_matrix = np.not_equal(b, b_est).float() # (batch,K)
    N_batch = error_matrix.shape[0]
    N_bits = error_matrix.shape[1]
    ber = sum(sum(error_matrix))/(N_batch*N_bits) 
    bler = sum((sum(error_matrix, axis=1)>0))/N_batch
    return ber, bler


def error_rate_onehot(d_est, b): # b -- (batch, K, 1)

    ind_est = torch.argmax(d_est, dim=1).squeeze(-1) # batch
    
    N_batch = b.size(0) 
    N_bits = b.size(1)
    b_est = dec2bin(ind_est, N_bits)                 # (batch, K)
    b = b.squeeze(-1)                                # (batch, K)
    
    error_matrix = np.not_equal(b, b_est).float() # batch,K
    ber = sum(sum(error_matrix))/(N_batch*N_bits) 
    bler = sum((torch.sum(error_matrix, dim=1)>0))/N_batch

    return ber, bler


# decimal to binary given bits, e.g., x=8 --> 00010 in 5 bits (Note. reverse representation)
def dec2bin(x, N_bits):
    mask = 2**torch.arange(N_bits) # .to(device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte() # add another axis, multiply bit seq, and denote it as binary