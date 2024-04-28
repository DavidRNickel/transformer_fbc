import torch

class FeedbackCode():
    def __init__(self, conf, transformer):
        self.device = conf.device
        self.batch_size = conf.batch_size
        self.N = conf.N
        self.K = conf.K
        self.transformer = transformer
        self.training = True

        # Power weighting-related parameters.
        self.weight_power = torch.nn.Parameter(torch.Tensor(self.param.N), requires_grad=True )
        self.weight_power.data.uniform_(1., 1.)
        self.weight_power_normalized = torch.sqrt(self.weigth_power**2 * (self.N)/torch.sum(self.weight_power**2))

        # Parameters for normalizing mean and variance of 
        self.mean_batch = torch.zeros(self.N)
        self.std_batch = torch.ones(self.N)
        self.mean_saved = torch.zeros(self.N)
        self.normalization_with_saved_data = False # True: inference w/ saved mean, var; False: calculate mean, var

    #
    # Do all the transmissions from the encoder side to the decoder side.
    def xmit_bits_from_encoder(self):
        xmitted_syms = None

        return xmitted_syms 

    #
    # Process the received symbols at the decoder side.
    def process_bits_at_decoder(self):
        # put the forward pass from the decoder here
        pass

    #
    # The following methods are taken from https://anonymous.4open.science/r/RCode1/main_RobustFeedbackCoding.ipynb
    # which is the code for the paper "Robust Non-Linear Feedback Coding via Power-Constrained Deep Learning".
    #

    #
    # Handle the power weighting on the transmit bits.
    def make_normalize_power_weights(self):
        pass

    #
    # Normalize the batch.
    def normalization(self, inputs, t_idx):
        mean_batch = torch.mean(inputs)
        std_batch = torch.std(inputs)
        if self.Training == True:
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
    # Convert the batch of bitstreams into a one-hot representation.
    def one_hot(self, bit_vec):
        # assumes bit_vec is (batch, K, 1) in shape
        bit_vec = bit_vec.squeeze(-1)

        ind = torch.arange(0,self.batch_size).repeat(self.batch_size,1).to(self.device)
        ind_vec = torch.sum(torch.mul(bit_vec,2**ind),axis=1).long()
        b_onehot = torch.zeros((self.batch_size, 2**self.K),dtype=int)
        for i in range(self.batch_size):
            b_onehot[i, ind_vec[i]] = 1
        
        return b_onehot
    