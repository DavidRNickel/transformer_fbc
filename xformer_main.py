import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import sys

from feedback_code_class import FeedbackCode
from config_class import Config


if __name__=='__main__':
    conf = Config()
    device = conf.device

    fbc = FeedbackCode(conf).to(device)

    # writer = None
    # if conf.use_tensorboard:
    #     log_folder = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    #     writer = SummaryWriter()
    # writer = SummaryWriter()

    num_epochs = 1 #conf.num_epochs 
    clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(fbc.parameters(), lr=.001, weight_decay=.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        fbc.train()
        bitstreams = torch.randint(0,2,(conf.batch_size, 1, conf.K)).to(device)
        H_real, H_imag = fbc.generate_split_channel_gains_rayleigh(shape=(conf.batch_size, conf.num_xmit_chans))
        H_prime = torch.cat((H_real,H_imag),axis=1).unsqueeze(1)
        feedback_info = -1 * torch.ones((conf.batch_size, 1, conf.N - 1)).to(device)
        knowledge_vectors = torch.cat((bitstreams, H_prime, feedback_info),axis=2).to(device) # do the positional encoding and concatenation at the same time

        optimizer.zero_grad()
        output = fbc(knowledge_vectors, H_real.squeeze(1), H_imag.squeeze(1))
        b = bitstreams.int().permute(0,2,1).squeeze(-1)
        b_one_hot = fbc.bits_to_one_hot(b).float()
        loss = loss_fn(output, b_one_hot)
