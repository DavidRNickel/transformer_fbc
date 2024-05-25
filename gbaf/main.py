import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import datetime
import sys
import pickle as pkl

from fbc_test import FeedbackCode
from config_class import Config
from timer_class import Timer


def test_model(model, conf):
    model.eval()
    batch_size = conf.batch_size
    device = conf.device
    ber = 0
    bler = 0
    pwr_avg = np.zeros((batch_size, conf.T))
    with torch.no_grad():
        for _ in range(conf.num_valid_epochs):
            bits = torch.randint(0, 2, (batch_size, conf.K)).to(device)
            b = bits.view(batch_size,-1,conf.M)
            output = model(b).view(bs*model.num_blocks, 2**conf.M)
            bit_estimates = model.one_hot_to_bits(output).bool().view(batch_size,-1).detach().clone().cpu().numpy().astype(np.bool_)
            ber_tmp, bler_tmp = model.calc_error_rates(bit_estimates, bits.detach().clone().cpu().numpy().astype(np.bool_))

            ber += ber_tmp
            bler += bler_tmp
            # pwr_avg += np.array(model.transmit_power_tracking).T

        ber /= conf.num_valid_epochs
        bler /= conf.num_valid_epochs
        # pwr_avg /= num_iters
    
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

    print('Setting up gradient descent stuff...')
    bs = conf.batch_size
    epoch_start = 0
    grad_clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(fbc.parameters(), lr=conf.optim_lr, weight_decay=conf.optim_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                  lr_lambda = lambda epoch: (1-epoch/conf.num_epochs))
    loss_fn = nn.CrossEntropyLoss()

    if conf.loadfile is not None:
        print('Loding checkpoint...')
        checkpoint = torch.load(conf.loadfile)
        fbc.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']

    bit_errors = []
    block_errors = []
    ctr = 0
    for epoch in range(epoch_start, conf.num_epochs):
        fbc.train()
        bitstreams = torch.randint(0, 2, (conf.batch_size, conf.K)).to(device) 

        b = bitstreams.view(bs,-1,conf.M)
        output = fbc(b)
        output = output.view(bs*fbc.num_blocks, 2**fbc.M)
        b_one_hot = fbc.bits_to_one_hot(b).float()
        loss = loss_fn(output, b_one_hot)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fbc.parameters(), grad_clip)
        L = loss.item()
        optimizer.step()
        scheduler.step()

        bit_estimates = fbc.one_hot_to_bits(output).bool().view(bs,-1)
        ber, bler = fbc.calc_error_rates(bit_estimates, bitstreams.bool())

        if conf.use_tensorboard:
            writer.add_scalar('loss/train/BER', ber, ctr)
            writer.add_scalar('loss/train/BLER', bler, ctr)
            writer.add_scalar('loss/train/loss', L, ctr)
            ctr += 1
    
        if epoch % 50 == 0:
            print(f'{epoch % conf.print_freq}', end=' ', flush=True)
        if epoch % conf.print_freq == 0:
            ber, bler, _ = test_model(model=fbc, conf=conf)
            bit_errors.append(ber)
            block_errors.append(bler)
            print(f'\nEpoch: {epoch}, Loss: {loss.item()}, BER: {ber}, BLER: {bler}\n')
            fbc.train()

        if conf.use_tensorboard:
            writer.add_scalar('loss/test/BER',ber,epoch)
            writer.add_scalar('loss/test/BLER',bler,epoch)

        if epoch % conf.save_freq == 0:
            nowtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save({'epoch' : epoch,
                        'model_state_dict' : fbc.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
                        'loss' : L},
                        f'{nowtime}.pt')
    
    print(f'ber: {np.array(bit_errors)}')
    print(f'bler: {np.array(block_errors)}')
    b = {'ber' : np.array(bit_errors), 'bler' : np.array(block_errors)}
    with open('test_results.pkl', 'wb') as f:
        pkl.dump(b,f)
