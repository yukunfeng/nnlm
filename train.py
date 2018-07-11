#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : training
"""

import argparse
import math
import time
import opts
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_logger
from utils.utils import word_ids_to_sentence
from nnlm import NNLM
import dataset


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    return opt


def train(opt, logger=None):
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
    model.rnn_encoder.embeddings.weight.data.copy_(TEXT.vocab.vectors)
    model.rnn_encoder.embeddings.weight.requries_grad =\
        opt.input_embeddings_trainable

    if opt.tied:
        model.out.weight = model.rnn_encoder.embeddings.weight

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(opt.lr))

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
                    prediction.view(-1, vocab_size),
                    target.view(-1))
                eval_total_loss += loss.item()
            return (eval_total_loss / batch_count)

    # Keep track of history ppl on val dataset
    val_ppls = []
    for epoch in range(1, int(opt.epoch) + 1):
        start_time = time.time()
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        for batch_count, batch_train in enumerate(train_iter, 1):
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

        # All xx_loss means loss per word on xx dataset
        train_loss = total_loss / batch_count

        # Doing validation
        val_loss = evaluation(val_iter)
        val_ppl = math.exp(val_loss)
        val_ppls.append(val_ppl)

        elapsed = time.time() - start_time
        start_time = time.time()

        if logger:
            logger.info('| epoch {:3d} | train_loss {:5.2f} '
                        '| val_ppl {:8.2f} | time {:5.1f}s'.format(
                            epoch,
                            train_loss,
                            val_ppl,
                            elapsed))

        # Saving model
        if epoch % opt.every_n_epoch_save == 0:
            if logger:
                logger.info("start to save model on {}".format(opt.save))
            with open(opt.save, 'wb') as save_fh:
                torch.save(model, save_fh)

    # Doing evaluation on test data
    test_loss = evaluation(test_iter)
    test_ppl = math.exp(test_loss)
    
    if logger:
        logger.info("test_ppl: {:5.1f}".format(test_ppl)) 


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
    logger.info("Start training...")
    logger.info(opt)
    train(opt, logger)
