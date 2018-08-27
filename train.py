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
from mlplm import MLPLM
from co_matrix import matrix_make


def adjust_learning_rate(optimizer, lr, decay):
    "adjust_learning_rate"
    #  lr = init_lr * (0.5 ** (epoch // every_n_epoch_decay))
    lr = lr * decay
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
    device = torch.device(opt.device)

    matrix = matrix_make(opt)
    vocab_size = 0
    word_dim = 0
    model = MLPLM(
        rnn_type=opt.rnn_type,
        bidirectional=opt.bidirectional,
        num_layers=opt.num_layers,
        vocab_size=vocab_size,
        word_dim=word_dim,
        hidden_size=word_dim * opt.window_len,
        dropout=opt.dropout
    ).to(device)

    model.embeddings.weight.data.copy_(TEXT.vocab.vectors)
    model.embeddings.weight.requires_grad = opt.update_inputemb

    if not opt.random_outemb:
        opt.out_emb_path = os.path.expanduser(opt.out_emb_path)
        out_emb = load_word_embedding(opt.out_emb_path)
        model.out.weight.data.copy_(out_emb)

    if opt.norm_out_emb:
        model.out.weight.data =\
            model.norm_tensor(model.out.weight.data).detach()

    model.out.weight.requires_grad = opt.update_out_emb

    if opt.tied:
        model.out.weight = model.embeddings.weight
    criterion = nn.CrossEntropyLoss()
    #  optimizer = optim.SGD(model.parameters(), lr=float(opt.lr))
    optimizer = optim.SGD(filter(
        lambda p: p.requires_grad,
        model.parameters()
    ), lr=float(opt.lr))

    def evaluation(data_iter):
        """do evaluation on data_iter
        return: average_word_loss"""
        model.eval()
        with torch.no_grad():
            eval_total_loss = 0
            count = 0
            for batch_count, batch_data in enumerate(data_iter, 1):
                text = batch_data.text.t().contiguous()
                target = batch_data.target.t().contiguous()
                for i in range(text.size(1) - opt.window_len + 1):
                    text_ = text[:, i:i+opt.window_len]
                    target_ = target[:, i+opt.window_len-1]
                    prediction = model(text_)
                    loss = criterion(
                        prediction, target_
                    )

                    eval_total_loss += loss.item()
                    count += 1
            return eval_total_loss / count

    # Keep track of history ppl on val dataset
    val_ppls = []
    last_val_ppl = 1000000
    for epoch in range(1, int(opt.epoch) + 1):
        start_time = time.time()
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        count = 0
        words = matrix.keys()
        for word in words:
            context_words = matrix[word]
            optimizer.zero_grad()
            prediction = model(text_)
            loss = criterion(
                prediction, target_
            )
            loss.backward()
            total_loss += loss.item()
            count += 1
            optimizer.step()

        # All xx_loss means loss per word on xx dataset
        train_loss = total_loss / count

        # Doing validation
        #  val_loss = evaluation_similarity(val_iter)
        #  val_ppl = val_loss
        val_loss = evaluation(val_iter)
        val_ppl = math.exp(val_loss)
        val_ppls.append(val_ppl)

        elapsed = time.time() - start_time
        start_time = time.time()

        if (last_val_ppl - val_ppl) <= 0.5:
            new_lr = adjust_learning_rate(
                optimizer, opt.lr, opt.decay
            )
            opt.lr = new_lr
            if logger:
                logger.info(f"learning rate has been changed to {new_lr}")
        last_val_ppl = val_ppl

        if logger:
            logger.info('| epoch {:3d} | train_loss {:5.2f} '
                        '| val_ppl {:8.5f} | time {:5.1f}s'.format(
                            epoch,
                            train_loss,
                            val_ppl,
                            elapsed))

        # Saving model
        if epoch % opt.every_n_epoch_save == 0:
            os.system(f"rm -f {opt.save}")
            if logger:
                logger.info("start to save model on {}".format(opt.save))
            opt.hidden_size = opt.word_dim * opt.window_len
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
    test_loss = evaluation(test_iter)
    test_ppl = math.exp(test_loss)
    if logger:
        logger.info("test_ppl: {:5.5f}".format(test_ppl))

    # saving output embeddings
    outemb_path = f"{opt.window_len}mlplen_{opt.epoch}epoch_outemb.txt"
    os.system(f"rm -f {outemb_path}")
    save_word_embedding(
        TEXT.vocab.itos,
        model.out.weight.data,
        outemb_path
    )


    # saving input embeddings
    #  save_word_embedding(
        #  TEXT.vocab.itos,
        #  "random_start_input_emb.txt"
    #  )


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
    logger.info("-------------")
    logger.info("Start training...")
    logger.info(opt)
    train(opt, logger)
