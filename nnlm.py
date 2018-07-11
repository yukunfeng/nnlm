#!/usr/bin/env python3
"""
Author      : Yukun Feng
Date        : 2018/07/03
Email       : yukunfg@gmail.com
Description : NN-based LM
"""

from __future__ import division
import torch
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

    def forward(self, src, target=None, lengths=None):
        _, memory_bank = self.rnn_encoder(src, lengths)
        if self.training:
            if target is None:
                raise Exception("target is none in training")
            memory_bank = memory_bank.view(-1, self.hidden_size)
            indexs = target.view(-1)
            target_embeddings = torch.index_select(self.rnn_encoder.embeddings.weight, 0, indexs)
            out = torch.bmm(
                memory_bank.view(-1, 1, self.hidden_size),
                target_embeddings.view(-1, self.hidden_size, 1)
            )
            out = out.view(-1)
            out = out.view(target.size(0), target.size(1))
            return out
        else:
            out = torch.mm(
                memory_bank.view(-1, self.hidden_size),
                self.rnn_encoder.embeddings.weight.t()
            )
            out = out.view(
                memory_bank.size(0),
                memory_bank.size(1),
                -1
            )
            return out

