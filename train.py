#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : training
"""

import argparse
import opts
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
    model = NNLM(
        rnn_type=opt.rnn_type,
        bidirectional=opt.bidirectional,
        num_layers=opt.num_layers,
        vocab_size=opt.vocab_size,
        word_dim=opt.word_dim,
        hidden_size=opt.word_dim,
        dropout=opt.dropout
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    exit(0)
    train_iter, test_iter, val_iter = dataset.create_lm_dataset()
    for batch_train in train_iter:
        optimizer.zero_grad()
        text, target = batch_train.text, batch_train.target
        prediction = model(text)
        loss = criterion(
            prediction.view(-1, model.vocab_size),
            target.view(-1)
        )
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
    logger.info("It's a test")
    train(opt)
