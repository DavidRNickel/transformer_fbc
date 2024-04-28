import torch
from torch import nn
import numpy as np

from feedback_code_class import FeedbackCode
from config_class import Config


if __name__=='__main__':
    conf = Config()
    device = conf.device
    fbc = FeedbackCode(conf).to(device)

    num_epochs = 1 #conf.num_epochs 
    clip = conf.grad_clip 
    optimizer = torch.optim.AdamW(fbc.parameters(), lr=.001, weight_decay=.01)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.tensor([[1,0,1,0],[1,1,1,1]]).to(device)
    y =fbc.bits_to_one_hot(x)
    print(fbc.one_hot_to_bits(y))

    # for epoch in range(num_epochs):
    #     bitstreams = np.random.randint(0,2,(conf.batch_size, 1, conf.K))
    #     H_real, H_imag = fbc.generate_split_channel_gains_rayleigh(shape=(conf.batch_size, 1, conf.num_xmit_chans))
    #     H_prime = np.concatenate((H_real,H_imag),axis=2)
    #     feedback_info = -1 * np.ones((conf.batch_size, 1, conf.N - 1))
    #     knowledge_vectors = torch.Tensor(np.concatenate((bitstreams, H_prime, feedback_info),axis=2)).to(device) # do the positional encoding and concatenation at the same time

    #     optimizer.zero_grad()
    #     output = fbc(knowledge_vectors, H_real.squeeze(1), H_imag.squeeze(1))
    #     b_one_hot = fbc.one_hot(torch.tensor(bitstreams).int().permute(0,2,1))
    #     print(b_one_hot.shape)
    #     # print(output.shape)
    #     # loss = loss_fn(output)