import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import datetime
import sys
import pickle as pkl

# from gtwc_class import GTWC
from lightcode_class import Lightcode
from config_class import Config
from timer_class import Timer


def test_model(test_data, model, conf):
    model.eval()
    num_test_streams = conf.num_valid_samps
    batch_size = conf.batch_size
    device = conf.device
    num_iters = (num_test_streams//batch_size)
    ber = 0
    ber_1 = 0
    ber_2 = 0
    bler = 0
    bler_1 = 0
    bler_2 = 0
    pwr_avg = np.zeros((batch_size, conf.N))
    with torch.no_grad():
        for i in range(num_iters):
            bits_1 = test_data['bits_1'][batch_size*i : batch_size*(i+1)].to(device)
            bits_2 = test_data['bits_2'][batch_size*i : batch_size*(i+1)].to(device)
            b1 = bits_1.view(batch_size,-1,conf.M)
            b2 = bits_2.view(batch_size,-1,conf.M)
            output_1, output_2 = model(b1, b2)
            output_1 = output_1.view(bs*model.num_blocks, 2**conf.M)
            output_2 = output_2.view(bs*model.num_blocks, 2**conf.M)
            bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(batch_size,-1).detach().clone().cpu().numpy().astype(np.bool_)
            bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(batch_size,-1).detach().clone().cpu().numpy().astype(np.bool_)
            ber_tmp_1, bler_tmp_1 = model.calc_error_rates(bit_estimates_1, bits_1.detach().clone().cpu().numpy().astype(np.bool_))
            ber_tmp_2, bler_tmp_2 = model.calc_error_rates(bit_estimates_2, bits_2.detach().clone().cpu().numpy().astype(np.bool_))

            ber_1 += ber_tmp_1
            ber_2 += ber_tmp_2
            bler_1 += bler_tmp_1
            bler_2 += bler_tmp_2
            ber += ber_tmp_1 + ber_tmp_2
            bler += bler_tmp_1 + bler_tmp_2
            
            # pwr_avg += np.array(model.transmit_power_tracking).T

        ber /= num_iters
        bler /= num_iters
        # pwr_avg /= num_iters
    
    return (ber, ber_1, ber_2), (bler, bler_1, bler_2), pwr_avg 


if __name__=='__main__':
    conf = Config()
    device = conf.device
    timer = Timer()

    # model = GTWC(conf).to(device)
    model = Lightcode(conf).to(device)

    if conf.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # Set up TensorBoard for logging purposes.
        writer = None
        if conf.use_tensorboard:
            log_folder = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            writer = SummaryWriter()
        writer = SummaryWriter()

    bs = conf.batch_size
    bitstreams_train_1 = torch.randint(0,2,(conf.num_training_samps, conf.K)).to(device)
    bitstreams_train_2 = torch.randint(0,2,(conf.num_training_samps, conf.K)).to(device)

    n_valid_samps = conf.num_valid_samps
    n_valid_iters = n_valid_samps // conf.batch_size
    test_bits_1 = torch.randint(0,2,(n_valid_samps, conf.K))
    test_bits_2 = torch.randint(0,2,(n_valid_samps, conf.K))
    test_data = {'bits_1' : test_bits_1,
                 'bits_2' : test_bits_2}

    num_epochs = conf.num_epochs 
    grad_clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.optim_lr, weight_decay=conf.optim_weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    bit_errors = []
    block_errors = []
    ctr = 0
    for epoch in range(num_epochs):
        model.train()
        losses = []
        for i in range(conf.num_iters_per_epoch):
            bitstreams_1 = bitstreams_train_1[bs*i:bs*(i+1)].int().to(device)
            bitstreams_2 = bitstreams_train_2[bs*i:bs*(i+1)].int().to(device)

            optimizer.zero_grad()
            b1, b2 = bitstreams_1.view(bs,-1,conf.M), bitstreams_2.view(bs,-1,conf.M)
            output_1, output_2 = model(b1, b2)
            output_1 = output_1.view(bs*model.num_blocks, 2**model.M)
            output_2 = output_2.view(bs*model.num_blocks, 2**model.M)
            b_one_hot_1 = model.bits_to_one_hot(b1).float()
            b_one_hot_2 = model.bits_to_one_hot(b2).float()
            loss = loss_fn(output_1, b_one_hot_1) + loss_fn(output_2, b_one_hot_2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            losses.append(L:=loss.item()) # only works in Python >= 3.11
            optimizer.step()

            bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(bs,-1)
            bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(bs,-1)
            ber_1, bler_1 = model.calc_error_rates(bit_estimates_1, bitstreams_1.bool())
            ber_2, bler_2 = model.calc_error_rates(bit_estimates_2, bitstreams_2.bool())
            ber = ber_1 + ber_2
            bler = bler_1 + bler_2

            if i % 100 == 0:
                print(f'Epoch (iter): {epoch} ({i}), Loss: {L}')

            if conf.use_tensorboard:
                writer.add_scalar('loss/train/BER', ber, ctr)
                writer.add_scalar('loss/train/BLER', bler, ctr)
                writer.add_scalar('loss/train/BER_1', ber_1, ctr)
                writer.add_scalar('loss/train/BLER_1', bler_1, ctr)
                writer.add_scalar('loss/train/BER_2', ber_2, ctr)
                writer.add_scalar('loss/train/BLER_2', bler_2, ctr)
                writer.add_scalar('loss/train/loss', L, ctr)
                ctr += 1
    
        ber_tup, bler_tup, _ = test_model(test_data=test_data, model=model, conf=conf)
        ber, ber_1, ber_2 = ber_tup
        bler, bler_1, bler_2 = bler_tup
        bit_errors.append(ber)
        block_errors.append(bler)

        scheduler.step()

        if conf.use_tensorboard:
            writer.add_scalar('loss/test/BER',ber,epoch)
            writer.add_scalar('loss/test/BLER',bler,epoch)
            writer.add_scalar('loss/test/BER_1',ber_1,epoch)
            writer.add_scalar('loss/test/BLER_1',bler_1,epoch)
            writer.add_scalar('loss/test/BER_2',ber_2,epoch)
            writer.add_scalar('loss/test/BLER_2',bler_2,epoch)

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
