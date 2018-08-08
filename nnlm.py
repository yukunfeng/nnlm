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
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, target=None, lengths=None, softmax=False):
        _, memory_bank = self.rnn_encoder(src, lengths)
        memory_bank = memory_bank.view(-1, self.hidden_size)

        # Norm vector
        # see https://pytorch.org/docs/master/_modules/torch/nn/functional.html#normalize
        memory_bank_norm = memory_bank.norm(
            2, 1, True
        ).clamp(min=1e-12).expand_as(memory_bank).detach()
        memory_bank = memory_bank / memory_bank_norm

        if softmax is False:
            if target is None:
                raise Exception("target is none in training")
            indexs = target.view(-1)

            # Norm vector
            target_embeddings = torch.index_select(self.out.weight, 0, indexs)
            target_embeddings_norm = target_embeddings.norm(
                2, 1, True
            ).clamp(min=1e-12).expand_as(target_embeddings).detach()
            target_embeddings = target_embeddings / target_embeddings_norm

            out = torch.bmm(
                memory_bank.view(-1, 1, self.hidden_size),
                target_embeddings.view(-1, self.hidden_size, 1)
            )
            out = out.view(-1)
            out = out.view(target.size(0), target.size(1))
            return out
        else:
            normed_out_weight = F.normalize(self.out.weight)
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

