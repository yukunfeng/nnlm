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
import os
import opts
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from nnlm import NNLM
import dataset


def adjust_learning_rate(optimizer, epoch, init_lr, every_n_epoch_decay):
    """Sets the learning rate to the initial
    LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // every_n_epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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
        opt, logger=logger
    )
    device = torch.device(opt.device)

    vocab_size = TEXT.vocab.vectors.size(0)
    word_dim = TEXT.vocab.vectors.size(1)

    if not opt.random_outemb:
        opt.out_emb_path = os.path.expanduser(opt.out_emb_path)
        out_emb = load_word_embedding(opt.out_emb_path)
        hidden_size = out_emb.size(1)

    model = NNLM(
        rnn_type=opt.rnn_type,
        bidirectional=opt.bidirectional,
        num_layers=opt.num_layers,
        vocab_size=vocab_size,
        word_dim=word_dim,
        hidden_size=hidden_size,
        dropout=opt.dropout
    ).to(device)

    model.rnn_encoder.embeddings.weight.data.copy_(TEXT.vocab.vectors)
    model.rnn_encoder.embeddings.weight.requires_grad = opt.update_inputemb

    if not opt.random_outemb:
        model.out.weight.data.copy_(out_emb)

    if opt.norm_out_emb:
        model.out.weight.data =\
            model.norm_tensor(model.out.weight.data).detach()

    model.out.weight.requires_grad = opt.update_out_emb

    if opt.tied:
        model.out.weight = model.rnn_encoder.embeddings.weight

    criterion = nn.CrossEntropyLoss()
    #  optimizer = optim.SGD(model.parameters(), lr=float(opt.lr))
    optimizer = optim.SGD(filter(
        lambda p: p.requires_grad,
        model.parameters()
    ), lr=float(opt.lr))

    best_val_loss = 1000000

    def evaluation_similarity(data_iter):
        """do evaluation on data_iter
        return: average_word_cosine"""
        model.eval()
        with torch.no_grad():
            eval_total_loss = 0
            for batch_count, batch_data in enumerate(data_iter, 1):
                text = batch_data.text.to(device)
                target = batch_data.target.to(device)
                loss = torch.mean(-model.forward_similarity(text, target))
                eval_total_loss += loss.item()
            return eval_total_loss / batch_count

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
            return eval_total_loss / batch_count

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
            # `clip_grad_norm` helps prevent the exploding gradient
            #  if epoch >= 14:
                #  torch.nn.utils.clip_grad_norm_(filter(
                    #  lambda p: p.requires_grad,
                    #  model.parameters()
                #  ), opt.clip)
                #  torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
            optimizer.step()

        # All xx_loss means loss per word on xx dataset
        train_loss = total_loss / batch_count

        # Doing validation
        #  val_loss = evaluation_similarity(val_iter)
        #  val_ppl = val_loss
        val_loss = evaluation(val_iter)
        val_ppl = math.exp(val_loss)
        val_ppls.append(val_ppl)

        elapsed = time.time() - start_time
        start_time = time.time()
        if logger:
            logger.info('| epoch {:3d} | train_loss {:5.2f} '
                        '| val_ppl {:8.5f} | time {:5.1f}s'.format(
                            epoch,
                            train_loss,
                            val_ppl,
                            elapsed))

        #  if epoch >= opt.decay_after_n_epoch and \
                #  epoch % opt.every_n_epoch_decay == 0:
            #  new_lr = adjust_learning_rate(
                #  optimizer, epoch,
                #  opt.lr, opt.every_n_epoch_decay
            #  )
            #  if logger:
                #  logger.info(f"learning rate has been changed to {new_lr}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            new_lr = adjust_learning_rate(
                optimizer, epoch,
                opt.lr, opt.every_n_epoch_decay
            )
            if logger:
                logger.info(f"learning rate has been changed to {new_lr}")



        # Saving model
        if epoch % opt.every_n_epoch_save == 0:
            if logger:
                logger.info("start to save model on {}".format(opt.save))
            opt.word_dim = word_dim
            opt.hidden_size = word_dim
            opt.vocab_size = vocab_size
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt': opt,
            }
            torch.save(
                save_dict,
                opt.save
            )

    # Doing evaluation on test data
    #  test_loss = evaluation_similarity(test_iter)
    #  test_ppl = test_loss
    test_loss = evaluation(test_iter)
    test_ppl = math.exp(test_loss)
    if logger:
        logger.info("test_ppl: {:5.5f}".format(test_ppl))

    # saving output embeddings
    save_word_embedding(
        TEXT.vocab.itos,
        model.out.weight.data,
        f"{opt.bptt_len}bptt_{opt.epoch}epoch_outemb.txt"
    )

    # saving input embeddings
    #  save_word_embedding(
        #  TEXT.vocab.itos,
        #  model.rnn_encoder.embeddings.weight.data,
        #  "random_start_input_emb.txt"
    #  )


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
    logger.info("-------------")
    logger.info("Start training...")
    logger.info(opt)
    train(opt, logger)
