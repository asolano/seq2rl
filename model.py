# coding: utf-8

import math
import torch
import torch.nn as nn


# See: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # FIXME d_model needs to be even for this encoding to work
        assert d_model % 2 == 0

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomTransformer(nn.Module):
    def __init__(self,
                 src_dim,
                 tgt_dim,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 max_seq_length
                 ):
        super(CustomTransformer, self).__init__()
        self.d_model = d_model

        # NOTE "Embeddings", i.e. adjust the dimensions from src -> d_model and tgt -> d_model
        self.fc_src = nn.Linear(src_dim, d_model)
        self.fc_tgt = nn.Linear(tgt_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_length)
        self.transformer = nn.Transformer(d_model,
                                          nhead,
                                          num_encoder_layers,
                                          num_decoder_layers,
                                          dim_feedforward,
                                          dropout)
        # Output mapping from d_model -> tgt
        self.fc_out = nn.Linear(d_model, tgt_dim)

    def forward(self,
                src,
                tgt,
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
                ):
        # Apply linear and positional encoding to sequences
        encoded_src = self.pos_enc.forward(self.fc_src(src) * math.sqrt(self.d_model))
        encoded_tgt = self.pos_enc.forward(self.fc_tgt(tgt) * math.sqrt(self.d_model))

        # Send to the model
        output = self.transformer(src=encoded_src,
                                  tgt=encoded_tgt,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        return self.fc_out(output)
