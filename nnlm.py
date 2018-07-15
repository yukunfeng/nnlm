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
import torch.nn.functional as F
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
        memory_bank = memory_bank.view(-1, self.hidden_size)
        memory_bank = F.normalize(memory_bank)
        if self.training:
            if target is None:
                raise Exception("target is none in training")
            indexs = target.view(-1)
            target_embeddings = torch.index_select(self.rnn_encoder.embeddings.weight, 0, indexs)
            target_embeddings = F.normalize(target_embeddings)
            out = torch.bmm(
                memory_bank.view(-1, 1, self.hidden_size),
                target_embeddings.view(-1, self.hidden_size, 1)
            )
            out = out.view(-1)
            out = out.view(target.size(0), target.size(1))
            return out
        else:
            normed_out_weight = F.normalize(self.rnn_encoder.embeddings.weight)
            out = torch.mm(
                memory_bank,
                normed_out_weight.t()
            )
            out = out.view(
                src.size(0),
                src.size(1),
                -1
            )
            return out

