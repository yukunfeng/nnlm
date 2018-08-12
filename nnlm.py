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
        self.out = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward_dot(self, src, target, lengths=None):
        _, memory_bank = self.rnn_encoder(src, lengths)
        memory_bank = memory_bank.view(-1, self.hidden_size)
        indexs = target.view(-1)
        target_embeddings = torch.index_select(self.out.weight, 0, indexs)
        out = torch.bmm(
            memory_bank.view(-1, 1, self.hidden_size),
            target_embeddings.view(-1, self.hidden_size, 1)
        )
        out = out.view(-1)
        out = out.view(target.size(0), target.size(1))
        return out

    def forward(self, src, lengths=None):
        _, memory_bank = self.rnn_encoder(src, lengths)
        out = self.out(memory_bank.view(-1, self.hidden_size))
        out = out.view(
            memory_bank.size(0),
            memory_bank.size(1),
            -1
        )
        return out

    def norm_tensor(self, tensor):
        """normalize tensor
        tensor: 2d tensor"""
        # see https://pytorch.org/docs/master/_modules/torch/nn/functional.html#normalize
        tensor_norm = tensor.norm(
            2, 1, True
        ).clamp(min=1e-12).expand_as(tensor).detach()
        tensor = tensor / tensor_norm
        return tensor

    def forward_similarity(self, src, target, lengths=None):
        """different from softmax. This is based on cosine similarity"""
        _, memory_bank = self.rnn_encoder(src, lengths)
        memory_bank = memory_bank.view(-1, self.hidden_size)

        # Norm vector
        memory_bank = self.norm_tensor(memory_bank)

        indexs = target.view(-1)
        # Norm vector
        target_embeddings = torch.index_select(self.out.weight, 0, indexs)
        target_embeddings = self.norm_tensor(target_embeddings)

        out = torch.bmm(
            memory_bank.view(-1, 1, self.hidden_size),
            target_embeddings.view(-1, self.hidden_size, 1)
        )
        out = out.view(-1)
        out = out.view(target.size(0), target.size(1))
        return out
