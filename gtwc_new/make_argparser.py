import argparse
from distutils.util import strtobool
import time
import torch

def make_parser():
    t = int(time.time())
    parser = argparse.ArgumentParser(description='Lightcode TWC Arguments')

    parser.add_argument('--use-tensorboard', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--loadfile', type=str, default=None, nargs='?', help='file containing saved torch model')
    parser.add_argument('--save-dir', type=str, default=None, nargs='?', help='directory to which everything is saved', required=True)
    parser.add_argument('--log-file', type=str, default=None, nargs='?', help='file to which logs are written', required=True)

    # need to set noise_pwr_ff/fb, know_vec_len, test_batch_size, num_training_samps
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--M', type=int, default=3)
    parser.add_argument('--T', type=int, default=9)
    parser.add_argument('--N', type=int, default=-1)
    
    parser.add_argument('--snr-ff', type=float, default=1, help='SNR1')
    parser.add_argument('--snr-fb', type=float, default=20, help='SNR2')

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', nargs='?', help='set \'cuda\' if available else \'cpu\'')

    parser.add_argument('--optim-lr', type=float, default=.001, help='optimizer learning rate')
    parser.add_argument('--optim-weight-decay', type=float, default=.01, help='decay for AdamW optimizer')
    parser.add_argument('--num-epochs', type=int, default=120)
    parser.add_argument('--num-validation-epochs', type=int, default=10)
    parser.add_argument('--num-test-epochs', type=int, default=int(1E5))
    parser.add_argument( '--batch-size', type=int, default=int(1E1))  
    parser.add_argument('--grad-clip', type=float, default=.5)

    return parser, t
