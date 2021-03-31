# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                # FIXME get mask from outside?
                device,
                src_mask=None,
                tgt_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None
                ):
        # Apply linear and positional encoding to sequences
        # FIXME pycharm complains about callable, needs explicit forward()
        encoded_src = self.pos_enc.forward(self.fc_src(src) * math.sqrt(self.d_model))
        encoded_tgt = self.pos_enc.forward(self.fc_tgt(tgt) * math.sqrt(self.d_model))

        # FIXME here or outside?
        src_mask = self.transformer.generate_square_subsequent_mask(encoded_src.shape[0]).to(device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(encoded_tgt.shape[0]).to(device)

        # Send to the model
        output = self.transformer(encoded_src,
                                  encoded_tgt,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask)

        return self.fc_out(output)


# PyTorch example code
class CustomRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, n_source, n_target, n_input, n_hidden, n_layers, dropout=0.5, tie_weights=False):
        super(CustomRNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Linear(in_features=n_source, out_features=n_input)
        # if rnn_type == 'LSTM':
        #     self.rnn = nn.LSTM(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers, dropout=dropout)
        # elif rnn_type == 'GRU':
        #     self.rnn = nn.GRU(input_size=n_input, hidden_size=n_hidden, num_layers=n_layers, dropout=dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_input, n_hidden, n_layers, dropout=dropout)
        else:
            if rnn_type == 'RNN_TANH':
                non_linearity = 'tanh'
            elif rnn_type == 'RNN_RELU':
                non_linearity = 'relu'
            else:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(n_input, n_hidden, n_layers, nonlinearity=non_linearity, dropout=dropout)

        self.decoder = nn.Linear(n_hidden, n_target)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if n_hidden != n_input:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, input, hidden):
        embedded = self.dropout(self.encoder(input))
        output, hidden = self.rnn.forward(embedded, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            n_directions = 1
            return (weight.new_zeros(self.n_layers * n_directions, batch_size, self.n_hidden),
                    weight.new_zeros(self.n_layers * n_directions, batch_size, self.n_hidden))
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.n_hidden)
