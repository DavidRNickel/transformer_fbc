import torch
from torch import nn
from torch.nn import functional as F


import numpy as np
import datetime
import sys
import pickle as pkl

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

            output = model(bits, noise_ff, noise_fb)
            # output = model(bits)
            bit_estimates = model.one_hot_to_bits(output).detach().clone().cpu().numpy().astype(np.bool_)
            ber_tmp, bler_tmp = model.calc_error_rates(bit_estimates, bits.squeeze(1).detach().clone().cpu().numpy().astype(np.bool_))

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

    if conf.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Set up TensorBoard for logging purposes.
        writer = None
        if conf.use_tensorboard:
            log_folder = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            writer = SummaryWriter()
        writer = SummaryWriter()

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
    optimizer = torch.optim.AdamW(fbc.parameters(), lr=.0005, weight_decay=.01)
    loss_fn = nn.CrossEntropyLoss()
    bit_errors = []
    block_errors = []
    for epoch in range(num_epochs):
        fbc.train()
        losses = []
        for i in range(conf.num_iters_per_epoch):
            bitstreams = torch.randint(0,2,(conf.batch_size, 1, conf.K)).to(device)
            feedback_info = -100 * torch.ones((conf.batch_size, 1, conf.N - 1)).to(device)

            optimizer.zero_grad()
            output = fbc(bitstreams)
            b = bitstreams.int().permute(0,2,1).squeeze(-1)
            b_one_hot = fbc.bits_to_one_hot(b).float()
            loss = loss_fn(output, b_one_hot)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fbc.parameters(), grad_clip)
            losses.append(L:=loss.item())
            optimizer.step()

            bit_estimates = fbc.one_hot_to_bits(output).bool()
            ber, bler = fbc.calc_error_rates(bit_estimates, bitstreams.squeeze(1).bool())

            if i % 100 == 0:
                print(f'Epoch (iter): {epoch} ({i}), Loss: {loss.item()}')

            if conf.use_tensorboard:
                ei = (epoch+1)*(i+1)
                writer.add_scalar('loss/train/BER', ber, ei)
                writer.add_scalar('loss/train/BLER', bler, ei)
                writer.add_scalar('loss/train/loss', L, ei)
    
        ber, bler, _ = test_model(test_data=test_data, model=fbc, conf=conf)
        bit_errors.append(ber)
        block_errors.append(bler)
        if conf.use_tensorboard:
            writer.add_scalar('loss/test/BER',ber,(epoch+1))
            writer.add_scalar('loss/test/BLER',bler,(epoch+1))

        print(f'\nEpoch Summary')
        print('====================================================')
        print(f'Epoch: {epoch}, Average loss: {np.mean(losses)}')
        print(f'BER: {ber:e}, BLER {bler:e}')
        print('====================================================\n')
    
    print(f'ber: {np.array(bit_errors)}')
    print(f'bler: {np.array(block_errors)}')
    b = {'ber' : np.array(bit_errors), 'bler' : np.array(block_errors)}
    with open('test_results.pkl', 'wb') as f:
        pkl.dump(b,f)
