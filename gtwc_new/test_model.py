import torch
import numpy as np

def test_model(N_bits, model, conf, show_progress_interval=None):
    bs = conf.batch_size
    num_iters = (N_bits//bs)
    model.eval()
    ber = 0
    ber_1 = 0
    ber_2 = 0
    bler = 0
    bler_1 = 0
    bler_2 = 0
    with torch.no_grad():
        for i in range(num_iters):
            if show_progress_interval is not None:
                if i % show_progress_interval == 0:
                    print(f'Iter: {i} (of {num_iters})')
            bits_1 = torch.randint(0,2, (bs, conf.K)).to(conf.device)
            bits_2 = torch.randint(0,2, (bs, conf.K)).to(conf.device)
            b1 = bits_1.view(bs,-1,conf.M)
            b2 = bits_2.view(bs,-1,conf.M)
            output_1, output_2 = model(b1, b2)
            output_1 = output_1.view(bs*model.num_blocks, 2**conf.M)
            output_2 = output_2.view(bs*model.num_blocks, 2**conf.M)
            bit_estimates_1 = model.one_hot_to_bits(output_1).bool().view(bs,-1).detach().clone().cpu().numpy().astype(np.bool_)
            bit_estimates_2 = model.one_hot_to_bits(output_2).bool().view(bs,-1).detach().clone().cpu().numpy().astype(np.bool_)
            ber_tmp_1, bler_tmp_1 = model.calc_error_rates(bit_estimates_1, bits_1.detach().clone().cpu().numpy().astype(np.bool_))
            ber_tmp_2, bler_tmp_2 = model.calc_error_rates(bit_estimates_2, bits_2.detach().clone().cpu().numpy().astype(np.bool_))

            ber_1 += ber_tmp_1
            ber_2 += ber_tmp_2
            bler_1 += bler_tmp_1
            bler_2 += bler_tmp_2
            ber += ber_tmp_1 + ber_tmp_2
            bler += bler_tmp_1 + bler_tmp_2
            
        ber /= num_iters; ber_1 /= num_iters; ber_2 /= num_iters; bler /= num_iters; bler_1 /= num_iters; bler_2 /= num_iters
    
    return (ber, ber_1, ber_2), (bler, bler_1, bler_2), None