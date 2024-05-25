import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from pos_enc_test import PositionalEncoding

def general_attention_network(dim_in, dim_out, dim_embed, d_model, activation, max_len, num_layers,
                              scaling_factor=4, n_heads=1, dropout=0, embed=None, dim_embed_out=None):

    if embed is None:
        if dim_embed_out is None:
            dim_embed_out = d_model
        embedding = nn.Sequential(nn.Linear(dim_in, dim_embed),
                                  activation,
                                  nn.Linear(dim_embed, dim_embed),
                                  activation,
                                  nn.Linear(dim_embed, dim_embed_out))
    else:
        embedding = embed
    
    pos_encoding = PositionalEncoding(d_model=d_model,
                                      dropout=dropout,
                                      max_len=max_len)

    attn_layer = TransformerEncoderLayer(d_model=d_model, 
                                         nhead=n_heads, 
                                         norm_first=True, 
                                         dropout=dropout, 
                                         dim_feedforward=scaling_factor*d_model,
                                         activation=activation,
                                         batch_first=True)

    attn_unit = TransformerEncoder(attn_layer, num_layers=num_layers)
    for name, param in attn_unit.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            nn.init.kaiming_uniform_(param)

    raw_output = nn.Linear(d_model, dim_out)
    nn.init.kaiming_uniform_(raw_output.weight)

    return embedding, pos_encoding, attn_unit, raw_output