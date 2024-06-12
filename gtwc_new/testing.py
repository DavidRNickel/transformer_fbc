import torch
import pickle as pkl
import sys
import os

from gtwc_class import GTWC
from test_model import test_model
from make_argparser import make_parser

if __name__=='__main__':
    parser, _ = make_parser()
    conf = parser.parse_args(sys.argv[1:])
    device = conf.device

    # Make necessary directories and files for logging
    os.makedirs(conf.save_dir, exist_ok=True) 
    orig_stdout = sys.stdout
    outfile = open(os.path.join(conf.save_dir, conf.log_file), 'w')
    sys.stdout=outfile

    # Make parameters that have to be calculated using other parameters
    conf.knowledge_vec_len = conf.M + 2*(conf.T-1) + 1 
    conf.d_model = 32
    conf.noise_pwr_ff = 10**(-conf.snr_ff/10)
    conf.use_belief_network = False
    conf.noise_pwr_fb = 10**(-conf.snr_fb/10)
    conf.test_batch_size = conf.batch_size
    conf.num_training_samps = int(1000 * conf.batch_size)
    conf.num_iters_per_epoch = conf.num_training_samps // conf.batch_size
    conf.num_layers_xmit = 2 
    conf.num_layers_belief = 2
    conf.num_layers_recv = 3
    conf.n_heads = 1
    conf.d_model = 32
    conf.scaling_factor = 4
    conf.dropout = 0.0

    model = GTWC(conf).to(conf.device)
    checkpoint = torch.load(conf.loadfile)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Testing...')
    bers, blers, _ = test_model(model, conf, show_progress_interval=10)

    print(f'BER: {bers}')
    print(f'BLER: {blers}')
    b = {'ber' : bers,
         'bler' : blers}
    test_no = int(1)
    with open(os.path.join(conf.save_dir, f'bler_{test_no}.pkl'), 'wb') as f:
        pkl.dump(b,f)

    sys.stdout = orig_stdout
    outfile.close()