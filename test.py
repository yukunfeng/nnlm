#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/10
Email       : yukunfg@gmail.com
Description : Test model. Here it doesn't mean to run on formal test data which has been done in
train.py once the training is over. It's used for case testing.
"""

import argparse
import math
import torch
import torch.nn as nn
from utils.utils import *
from nnlm import NNLM
import dataset


def load_nnlm(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    opt = checkpoint['opt']
    TEXT, train_iter, test_iter, val_iter = dataset.create_lm_dataset(
        resources_dir=opt.resources_dir,
        vector_type=opt.vector_type,
        batch_size=opt.batch_size,
        bptt_len=opt.bptt_len,
        device=opt.device,
        logger=None
    )
    device = torch.device(opt.device)

    model = NNLM(
        rnn_type=opt.rnn_type,
        bidirectional=opt.bidirectional,
        num_layers=opt.num_layers,
        vocab_size=opt.vocab_size,
        word_dim=opt.word_dim,
        hidden_size=opt.hidden_size,
        dropout=opt.dropout
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, opt


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    return opt


def main():
    model, opt = load_nnlm("./nnlm.model")
    TEXT, train_iter, test_iter, val_iter = dataset.create_lm_dataset(
        resources_dir=opt.resources_dir,
        vector_type=opt.vector_type,
        batch_size=opt.batch_size,
        bptt_len=opt.bptt_len,
        device=opt.device,
        logger=None
    )
    device = torch.device(opt.device)

    criterion = nn.CrossEntropyLoss()
    #  optimizer = optim.SGD(model.parameters(), lr=float(opt.lr))

    def evaluation(data_iter):
        """do evaluation on data_iter
        return: average_word_loss"""
        model.eval()
        with torch.no_grad():
            eval_total_loss = 0
            for batch_count, batch_data in enumerate(data_iter, 1):
                text = batch_data.text.to(device)
                target = batch_data.target.to(device)
                prediction = model(text)
                loss = criterion(
                    prediction.view(-1, opt.vocab_size),
                    target.view(-1))
                eval_total_loss += loss.item()
            return eval_total_loss / batch_count

    test_loss = evaluation(test_iter)
    test_ppl = math.exp(test_loss)
    print(test_ppl)


if __name__ == "__main__":
    main()
