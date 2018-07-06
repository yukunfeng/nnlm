#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : training
"""

import argparse
import opts
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_logger
from nnlm import NNLM
import dataset


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    #  torch.manual_seed(opt.seed)

    return opt


def train(opt):
    """training given opt"""
    TEXT, train_iter, test_iter, val_iter = dataset.create_lm_dataset(
        resources_dir=opt.resources_dir,
        vector_type=opt.vector_type,
        batch_size=opt.batch_size,
        bptt_len=opt.bptt_len,
        device=opt.device
    )
    device = torch.device(opt.device)

    vocab_size = TEXT.vocab.vectors.size(0)
    word_dim = TEXT.vocab.vectors.size(1)
    model = NNLM(
        rnn_type=opt.rnn_type,
        bidirectional=opt.bidirectional,
        num_layers=opt.num_layers,
        vocab_size=vocab_size,
        word_dim=word_dim,
        hidden_size=word_dim,
        dropout=opt.dropout
    ).to(device)
    model.rnn_encoder.embeddings.weight = nn.Parameter(TEXT.vocab.vectors)
    model.rnn_encoder.embeddings.weight.requries_grad =\
        opt.input_embeddings_trainable

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    total_loss = 0
    for batch_train in train_iter:
        optimizer.zero_grad()
        text = batch_train.text.to(device)
        target = batch_train.target.to(device)
        prediction = model(text)
        loss = criterion(
            prediction.view(-1, vocab_size),
            target.view(-1)
        )
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        print(loss.item())


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
    logger.info("It's a test")
    train(opt)
