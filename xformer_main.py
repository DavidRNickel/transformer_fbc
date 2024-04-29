import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import sys

# from feedback_code_class import FeedbackCode
from fbc_test import FeedbackCode
from config_class import Config
from timer_class import Timer


def test_model(test_data, model, conf):
    model.eval()
    num_test_streams = conf.num_valid_samps
    batch_size = conf.batch_size
    device = conf.device
    num_iters = (num_test_streams//batch_size)
    ber = 0
    bler = 0
    pwr_avg = np.zeros((batch_size, conf.N))
    with torch.no_grad():
        for i in range(num_iters):
            bits = test_data['bits'][batch_size*i : batch_size*(i+1)].unsqueeze(1).to(device)
            noise_ff = test_data['noise_ff'][batch_size*i : batch_size*(i+1)].to(device)
            noise_fb = test_data['noise_fb'][batch_size*i : batch_size*(i+1)].to(device)
            H_real, H_imag = fbc.generate_split_channel_gains_rayleigh(shape=(conf.batch_size, conf.num_xmit_chans))

            output = model(bits, H_real, H_imag, noise_ff, noise_fb)
            ber_tmp, bler_tmp = model.calc_error_rates(output, bits)

            ber += ber_tmp
            bler += bler_tmp
            pwr_avg += np.array(model.transmit_power_tracking).T

        ber /= num_iters
        bler /= num_iters
        pwr_avg /= num_iters
    
    return ber, bler, pwr_avg 


if __name__=='__main__':
    conf = Config()
    device = conf.device
    timer = Timer()

    fbc = FeedbackCode(conf).to(device)

    # Set up TensorBoard for logging purposes.
    # writer = None
    # if conf.use_tensorboard:
    #     log_folder = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    #     writer = SummaryWriter()
    # writer = SummaryWriter()

    n_valid_samps = conf.num_valid_samps
    n_valid_iters = n_valid_samps // conf.batch_size
    test_bits = torch.randint(0,2,(n_valid_samps, conf.K))
    test_noise_ff = np.sqrt(conf.noise_pwr_ff) * torch.randn(size=(n_valid_samps, conf.N))
    test_noise_fb = np.sqrt(conf.noise_pwr_fb) * torch.randn(size=(n_valid_samps, conf.N))
    test_data = {'bits' : test_bits,
                 'noise_ff' : test_noise_ff,
                 'noise_fb' : test_noise_fb}

    num_epochs = conf.num_epochs 
    grad_clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(fbc.parameters(), lr=.001, weight_decay=.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        fbc.train()
        losses = []
        for i in range(conf.num_iters_per_epoch):
            bitstreams = torch.randint(0,2,(conf.batch_size, 1, conf.K)).to(device)
            H_real, H_imag = fbc.generate_split_channel_gains_rayleigh(shape=(conf.batch_size, conf.num_xmit_chans))
            feedback_info = -1 * torch.ones((conf.batch_size, 1, 2*conf.N - 2)).to(device)

            optimizer.zero_grad()
            output = fbc(bitstreams, H_real, H_imag)
            b = bitstreams.int().permute(0,2,1).squeeze(-1)
            b_one_hot = fbc.bits_to_one_hot(b).float()
            loss = loss_fn(output, b_one_hot)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fbc.parameters(), grad_clip)
            losses.append(loss.item())
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch (iter): {epoch} ({i}), Loss: {loss.item()}')
    
        print(f'\nEpoch Summary')
        print('====================================================')
        print(f'Epoch: {epoch}, Average loss: {np.mean(losses)}')
        print('====================================================\n')

        ber_val = test_model(test_data=test_data, 
                             model=fbc,
                             conf=conf)