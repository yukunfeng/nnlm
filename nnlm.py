#!/usr/bin/env python3
"""
Author      : Yukun Feng
Date        : 2018/07/03
Email       : yukunfg@gmail.com
Description : NN-based LM
"""

from __future__ import division
import torch.nn as nn
from rnn_encoder import RNNEncoder


class NNLM(nn.Module):
    """NNLM class"""
    def __init__(self, rnn_type, bidirectional,
                 num_layers, vocab_size, word_dim, hidden_size, dropout=0.0):
        super(NNLM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_encoder = RNNEncoder(
            rnn_type,
            bidirectional,
            num_layers,
            vocab_size,
            word_dim,
            hidden_size,
            dropout=0.0
        )
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, lengths=None):
        _, memory_bank = self.rnn_encoder(src, lengths)
        out = self.out(memory_bank.view(-1, self.hidden_size))
        out = out.view(
            memory_bank.size(0),
            memory_bank.size(1),
            memory_bank.size(2)
        )
        return out

