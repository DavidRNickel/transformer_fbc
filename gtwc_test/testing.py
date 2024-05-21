import torch
import pickle as pkl

from gtwc_class import GTWC
from config_class import *
from test_model import test_model

if __name__=='__main__':
    conf = Config()
    device = conf.device
    gtwc = GTWC(conf).to(device)
    checkpoint = torch.load(conf.loadfile)
    print('Loading checkpoint...')
    gtwc.load_state_dict(checkpoint['model_state_dict'])

    print('Testing...')
    N = int(1E8)
    bers, blers, _ = test_model(N, gtwc, conf, show_progress_interval=1000)

    print(f'BER: {bers}')
    print(f'BLER: {blers}')
    b = {'ber' : bers, 'bler' : blers}

    with open(f'{conf.loadfile[:-3]}_bler.pkl','wb') as f:
        pkl.dump(b,f)