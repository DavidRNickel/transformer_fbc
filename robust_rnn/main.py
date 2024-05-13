import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from module import *

class Feedback_Code(torch.nn.Module):
    def __init__(self, param):
        super(Feedback_Code, self).__init__()
        
        # import parameter
        self.param        = param
        if self.param.decoder_bidirection == True:
            self.decoder_bi = 2 # bi-direction
        else:
            self.decoder_bi = 1 # uni-direction

        # input_type (bit vector, one-hot vector) -- Encoder
        if self.param.input_type == 'bit_vector':
            self.num_input = self.param.N_bits
        elif self.param.input_type == 'one_hot_vector':
            self.num_input = 2**self.param.N_bits
        
        # output_type (bits, one-hot vector) -- Decoder
        if self.param.output_type == 'bit_vector':
            self.num_output = self.param.N_bits
        elif self.param.output_type == 'one_hot_vector':
            self.num_output = 2**self.param.N_bits

        # encoder RNN
        self.encoder_RNN   = torch.nn.GRU(self.num_input + 1, self.param.encoder_N_neurons, num_layers = self.param.encoder_N_layers, 
                                          bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.encoder_linear = torch.nn.Linear(self.param.encoder_N_neurons, 1)

        # power weights
        self.weight_power = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use), requires_grad = True )
        self.weight_power.data.uniform_(1.0, 1.0) # all 1
        self.weight_power_normalized = torch.sqrt(self.weight_power**2 *(self.param.N_channel_use)/torch.sum(self.weight_power**2))

        # decoder 
        self.decoder_RNN = torch.nn.GRU(1, self.param.decoder_N_neurons, num_layers = self.param.decoder_N_layers, 
                                        bias=True, batch_first=True, dropout=0, bidirectional= self.param.decoder_bidirection) 
        self.decoder_linear = torch.nn.Linear(self.decoder_bi*self.param.decoder_N_neurons, self.num_output) # 100,10

        # attention type
        if self.param.attention_type==5:  # bi-directional --> 2N weights
            self.weight_merge = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use,2), requires_grad = True ) 
            self.weight_merge.data.uniform_(1.0, 1.0) # all 1
            # Normalization
            self.weight_merge_normalized_fwd = torch.sqrt(self.weight_merge[:,0]**2 *(self.param.N_channel_use)/torch.sum(self.weight_merge[:,0]**2)) 
            self.weight_merge_normalized_bwd  = torch.sqrt(self.weight_merge[:,1]**2 *(self.param.N_channel_use)/torch.sum(self.weight_merge[:,1]**2))
        
        if self.param.attention_type== 4: # uni-directional --> N weights
            self.weight_merge = torch.nn.Parameter(torch.Tensor(self.param.N_channel_use),requires_grad = True )
            self.weight_merge.data.uniform_(1.0, 1.0) # all 1
            # Normalization
            self.weight_merge_normalized  = torch.sqrt(self.weight_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight_merge**2))
        
        # Parameters for normalization (mean and variance)
        self.mean_batch = torch.zeros(self.param.N_channel_use) 
        self.std_batch = torch.ones(self.param.N_channel_use)
        self.mean_saved = torch.zeros(self.param.N_channel_use)
        self.std_saved = torch.ones(self.param.N_channel_use)
        self.normalization_with_saved_data = False   # True: inference with saved mean/var, False: calculate mean/var

    def decoder_activation(self, inputs):
        if self.param.output_type == 'bit_vector':
            return torch.sigmoid(inputs) # training with binary cross entropy
        elif self.param.output_type == 'one_hot_vector':
            return inputs # Note. softmax function is applied in "F.cross_entropy" function
    
    # Convert `bit vector' to 'one-hot vector'
    def one_hot(self, bit_vec):
        bit_vec = bit_vec.view(parameter.batch_size, parameter.N_bits)
        N_batch = bit_vec.size(0) # batch_size
        N_bits = bit_vec.size(1)  # N_bits=K

        ind = torch.arange(0,N_bits).repeat(N_batch,1) 
        ind = ind.to(device)
        ind_vec = torch.sum( torch.mul(bit_vec, 2**ind), axis=1 ).long()
        bit_onehot = torch.zeros((N_batch, 2**N_bits), dtype=int)
        for ii in range(N_batch):
            bit_onehot[ii, ind_vec[ii]]=1 # one-hot vector
        return bit_onehot 
        
    def normalization(self, inputs, t_idx):
        if self.training: # During training
            mean_batch = torch.mean(inputs)
            std_batch  = torch.std(inputs)
            outputs   = (inputs - mean_batch)/std_batch
        else: 
            if self.normalization_with_saved_data: # During inference
                outputs   = (inputs - self.mean_saved[t_idx])/self.std_saved[t_idx]
            else: # During validation
                mean_batch = torch.mean(inputs)
                std_batch  = torch.std(inputs)
                self.mean_batch[t_idx] = mean_batch
                self.std_batch[t_idx] = std_batch
                outputs   = (inputs - mean_batch)/std_batch
        return outputs


    def forward(self, b, noise1, noise2):

        # Normalize power weights
        self.weight_power_normalized  = torch.sqrt(self.weight_power**2 *(self.param.N_channel_use)/torch.sum(self.weight_power**2))

        # Encoder input
        if self.param.input_type == 'bit_vector':
            I = b 
        elif self.param.input_type == 'one_hot_vector':
            b_hot = self.one_hot(b).to(device)
            I = b_hot 
        
        for t in range(self.param.N_channel_use): # timesteps
            # Encoder
            if t == 0: # 1st timestep
                input_total        = torch.cat([I.view(self.param.batch_size, 1, self.num_input), 
                                               torch.zeros((self.param.batch_size, 1, 1)).to(device)], dim=2) 
                ### input_total -- (batch,1, num_input + 1) 
                x_t_after_RNN, s_t_hidden  = self.encoder_RNN(input_total)
                ### x_t_after_RNN -- (batch, 1, hidden)
                ### s_t -- (layers, batch, hidden)
                x_t_tilde =   torch.tanh(self.encoder_linear(x_t_after_RNN))   
                
            else: # 2-30nd timestep
                input_total        = torch.cat([I.view(self.param.batch_size, 1, self.num_input), z_t], dim=2) 
                x_t_after_RNN, s_t_hidden  = self.encoder_RNN(input_total, s_t_hidden)
                x_t_tilde =   torch.tanh(self.encoder_linear(x_t_after_RNN))
            
            # Power control layer: 1. Normalization, 2. Power allocation
            x_t_norm = self.normalization(x_t_tilde,t).view(self.param.batch_size, 1, 1)
            x_t  = x_t_norm * self.weight_power_normalized[t] 
            
            # Forward transmission
            y_t = x_t + noise1[:,t,:].view(self.param.batch_size, 1, 1)
            
            # Feedback transmission
            z_t = y_t + noise2[:,t,:].view(self.param.batch_size, 1, 1)
            
            # Concatenate values along time t
            if t == 0:
                x_norm_total = x_t_norm
                x_total = x_t
                y_total = y_t
                z_total = z_t
            else:
                x_norm_total = torch.cat([x_norm_total, x_t_norm], dim=1) 
                x_total = torch.cat([x_total, x_t ], dim = 1) # In the end, (batch, N, 1)
                y_total = torch.cat([y_total, y_t ], dim = 1) 
                z_total = torch.cat([z_total, z_t ], dim = 1) 
     
        # Decoder
        # Normalize attention weights
        if parameter.attention_type== 4:
            self.weight_merge_normalized  = torch.sqrt(self.weight_merge**2 *(self.param.N_channel_use)/torch.sum(self.weight_merge**2)) 
        if parameter.attention_type== 5:
            self.weight_merge_normalized_fwd  = torch.sqrt(self.weight_merge[:,0]**2 *(self.param.N_channel_use)/torch.sum(self.weight_merge[:,0]**2)) # 30
            self.weight_merge_normalized_bwd  = torch.sqrt(self.weight_merge[:,1]**2 *(self.param.N_channel_use)/torch.sum(self.weight_merge[:,1]**2))
            
        decoder_input = y_total
        r_hidden, _  = self.decoder_RNN(decoder_input) # (batch, N, bi*hidden_size)
        
        # Option 1. Only the N-th timestep
        if parameter.attention_type== 1:
            output     = self.decoder_activation(self.decoder_linear(r_hidden)) #(batch,N,bi*hidden)-->(batch,N,num_output)
            output_last = output[:,-1,:].view(self.param.batch_size,-1,1) # (batch,num_output,1)
        
        # Option 2. Merge the "last" outputs of forward/backward RNN
        if parameter.attention_type== 2:
            r_backward = r_hidden[:,0,self.param.decoder_N_neurons:] # Output at the 1st timestep of reverse RNN 
            r_forward = r_hidden[:,-1,:self.param.decoder_N_neurons] # Output at the N-th timestep of forward RNN
            r_concat = torch.cat([r_backward, r_forward ], dim = 1) 
            output = self.decoder_activation(self.decoder_linear(r_concat)) # (batch,num_output)
            output_last = output.view(self.param.batch_size,-1,1) # (batch,num_output,1)
            
        # Option 3. Sum over all timesteps
        if parameter.attention_type== 3:
            output     = self.decoder_activation(self.decoder_linear(r_hidden)) 
            output_last = torch.sum(output, dim=1).view(self.param.batch_size,-1,1) # (batch,num_output,1)

        # Option 4. Attention mechanism (N weights)
        if parameter.attention_type== 4:
            r_concat = torch.tensordot(r_hidden, self.weight_merge_normalized, dims=([1], [0])) # (batch,bi*hidden_size)
            output = self.decoder_activation(self.decoder_linear(r_concat)) 
            output_last = output.view(self.param.batch_size,-1,1) # (batch,num_output,1)
            
        # Option 5. Attention mechanism (2N weights) for forward/backward
        if parameter.attention_type== 5:
            r_hidden_forward = r_hidden[:,:,:self.param.decoder_N_neurons]  # (batch,num_output,hidden_size)
            r_hidden_backward = r_hidden[:,:,self.param.decoder_N_neurons:] # (batch,num_output,hidden_size)
            r_forward_weighted_sum = torch.tensordot(r_hidden_forward, self.weight_merge_normalized_fwd, dims=([1], [0]))  # (batch,hidden_size)
            r_backward_weighted_sum = torch.tensordot(r_hidden_backward, self.weight_merge_normalized_bwd, dims=([1], [0]))         # (batch,hidden_size)
            r_concat = torch.cat([r_forward_weighted_sum, r_backward_weighted_sum], dim = 1) 
            output = self.decoder_activation(self.decoder_linear(r_concat)) 
            output_last = output.view(self.param.batch_size,-1,1) # (batch,num_output,1)

        self.x = x_total                    # (batch,N,1)

        return output_last
# Convert the `bit vector' with (batch,K,1) to 'one hot vector' with (batch,2^K)
def one_hot(bit_vec):
    bit_vec = bit_vec.squeeze(-1)  # (batch, K)
    N_batch = bit_vec.size(0) 
    N_bits = bit_vec.size(1)

    ind = torch.arange(0,N_bits).repeat(N_batch,1) # (batch, K)
    ind = ind.to(device)
    ind_vec = torch.sum( torch.mul(bit_vec,2**ind), axis=1).long() # batch
    b_onehot = torch.zeros((N_batch, 2**N_bits), dtype=int)
    for ii in range(N_batch):
        b_onehot[ii, ind_vec[ii]]=1 # one-hot vector
    return b_onehot
# Test
def test_RNN(N_test): 

    # Generate test data
    bits_test     = torch.randint(0, 2, (N_test, parameter.N_bits, 1)) 
    noise1_test  = sigma1*torch.randn((N_test, parameter.N_channel_use,1))
    noise2_test   = sigma2*torch.randn((N_test, parameter.N_channel_use,1))
    
    model.eval() # model.training() becomes False
    N_iter = (N_test//parameter.batch_size) # N_test should be multiple of batch_size
    ber=0
    bler=0
    power_avg = np.zeros((parameter.batch_size, parameter.N_channel_use ,1))
    with torch.no_grad():
        for i in range(N_iter):
            bits = bits_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) # batch, K,1
            noise1 = noise1_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1) # batch, N,1
            noise2 = noise2_test[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1) # batch, N,1

            bits = bits.to(device)
            noise1 = noise1.to(device)
            noise2 = noise2.to(device)

            # Forward pass
            output = model(bits, noise1, noise2)

            if parameter.output_type == 'bit_vector':
                ber_tmp, bler_tmp = error_rate_bitvector(output.cpu(), bits.cpu())
            elif parameter.output_type == 'one_hot_vector':
                ber_tmp, bler_tmp = error_rate_onehot(output.cpu(), bits.cpu())

            ber = ber + ber_tmp
            bler = bler + bler_tmp
            # Power
            signal = model.x.cpu().detach().numpy()
            power_avg += signal**2 
            
        ber  = ber/N_iter
        bler = bler/N_iter
        power_avg = power_avg/N_iter

    return ber, bler, power_avg
# model setup
parameter = params()
use_cuda = parameter.use_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model = Feedback_Code(parameter).to(device)
else:
    model = Feedback_Code(parameter)

print(model)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameter.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
# Generate training data
SNR1 = 1              # forward SNR in dB
np1 = 10**(-SNR1/10)   # noise power1 -- Assuming signal power is set to 1
sigma1 = np.sqrt(np1)
np2_dB = -20           # noise power2 in dB  
np2 = 10**(np2_dB/10)
sigma2 = np.sqrt(np2)

# Training set: tuples of (stream, noise1, noise 2)
N_train = int(1e7)  # number of training set
bits_train     = torch.randint(0, 2, (N_train, parameter.N_bits, 1))
noise1_train  = sigma1*torch.randn((N_train, parameter.N_channel_use, 1)) 
noise2_train   = sigma2*torch.randn((N_train, parameter.N_channel_use, 1)) 

# Validation
N_validation = int(1e5)

print('np1: ', np1)
print('np2: ', np2)
# Training
num_epoch = 100
clipping_value = 1
 
print('Before training ')
print('weight_power: ', model.weight_power_normalized.cpu().detach().numpy().round(3))
if parameter.attention_type==4:
    print('weight_merge: ', model.weight_merge_normalized.cpu().detach().numpy().round(3))
if parameter.attention_type==5:
    print('weight_merge_fwd: ', model.weight_merge_normalized_fwd.cpu().detach().numpy().round(3))
    print('weight_merge_bwd: ', model.weight_merge_normalized_bwd.cpu().detach().numpy().round(3))
print()

for epoch in range(num_epoch):

    model.train() # model.training() becomes True
    loss_training = 0
    
    N_iter = (N_train//parameter.batch_size)
    for i in range(N_iter):
        bits = bits_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) 
        noise1 = noise1_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1)
        noise2 = noise2_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1)

        bits   = bits.to(device)
        noise1 = noise1.to(device)
        noise2 = noise2.to(device)

        # forward pass
        optimizer.zero_grad() 
        output = model(bits, noise1, noise2)

        # Define loss according to output type
        if parameter.output_type == 'bit_vector':
            loss = F.binary_cross_entropy(output, b) 
        elif parameter.output_type == 'one_hot_vector':
            b_hot =  one_hot(bits).view(parameter.batch_size, 2**parameter.N_bits, 1) # (batch,2^K,1)
            loss = F.cross_entropy(output.squeeze(-1), torch.argmax(b_hot,dim=1).squeeze(-1).to(device))

        # training
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        loss_training += loss.item()
        optimizer.step()
        
        if i % 100 == 0:
            print('Epoch: {}, Iter: {} out of {}, Loss: {:.4f}'.format(epoch, i, N_iter, loss.item()))
            model.eval()
            b,l,_ = test_RNN(10000)
            print(f'Intermittent: BER: {b:e}, BLER: {l:e}\n')
            model.train()

    # Summary of each epoch
    print('Summary: Epoch: {}, lr: {}, Average loss: {:.4f}'.format(epoch, optimizer.param_groups[0]['lr'], loss_training/N_iter) )

    scheduler.step() # reduce learning rate
    
    print('weight_power', model.weight_power_normalized.cpu().detach().numpy())
    if parameter.attention_type==4:
        print('weight_merge: ', model.weight_merge_normalized.cpu().detach().numpy().round(3))
    if parameter.attention_type==5:
        print('weight_merge_fwd: ', model.weight_merge_normalized_fwd.cpu().detach().numpy().round(3))
        print('weight_merge_bwd: ', model.weight_merge_normalized_bwd.cpu().detach().numpy().round(3))
    print()
    
    # Validation
    ber_val, bler_val, _ = test_RNN(N_validation)
    print('Ber:  ', float(ber_val))
    print('Bler: ', float(bler_val))
    print()

# Calculate mean/var with training data
model.eval()   # model.training() becomes False
N_iter = N_train//parameter.batch_size
mean_train = torch.zeros(parameter.N_channel_use)
std_train  = torch.zeros(parameter.N_channel_use)
mean_total = 0
std_total = 0

with torch.no_grad():
    for i in range(N_iter):
        bits   = bits_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_bits,1) 
        noise1 = noise1_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1)
        noise2 = noise2_train[parameter.batch_size*i:parameter.batch_size*(i+1),:,:].view(parameter.batch_size, parameter.N_channel_use,1)

        bits   = bits.to(device)
        noise1 = noise1.to(device)
        noise2 = noise2.to(device)
        
        output = model(bits, noise1, noise2)
        mean_total += model.mean_batch
        std_total  += model.std_batch
        if i%100==0: print(i)
        
mean_train = mean_total/N_iter
std_train = std_total/N_iter
print('Mean: ',mean_train)
print('std : ',std_train)
# Inference stage
N_inference = int(1e8) 
N_small = int(1e5) # In case that N_inference is very large, we divide into small chunks
N_iter  = N_inference//N_small

model.normalization_with_saved_data = True
model.mean_saved = mean_train
model.std_saved  = std_train

ber_sum  = 0
bler_sum = 0

for ii in range(N_iter):
    ber_tmp, bler_tmp, _ = test_RNN(N_small)
    ber_sum += ber_tmp
    bler_sum += bler_tmp
    if ii%50==0: 
        print('Iter: {} out of {}'.format(ii, N_iter))
        print('Ber:  ', float(ber_sum/(ii+1)))
        print('Bler: ', float(bler_sum/(ii+1)))

ber_inference  = ber_sum/N_iter
bler_inference = bler_sum/N_iter
    
print()
print('Ber:  ', float(ber_inference))
print('Bler: ', float(bler_inference))
# End