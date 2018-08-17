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
from utils.utils import *
from mlplm import MLPLM
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
        device=opt.device,
        logger=logger
    )
    device = torch.device(opt.device)

    vocab_size = TEXT.vocab.vectors.size(0)
    #  word_dim = TEXT.vocab.vectors.size(1)
    model = MLPLM(
        rnn_type=opt.rnn_type,
        bidirectional=opt.bidirectional,
        num_layers=opt.num_layers,
        vocab_size=vocab_size,
        word_dim=opt.word_dim,
        hidden_size=opt.word_dim * opt.window_len,
        dropout=opt.dropout
    ).to(device)
    model.embeddings.weight.data.copy_(TEXT.vocab.vectors)
    model.embeddings.weight.requires_grad = opt.update_inputemb

    if not opt.random_outemb:
        out_emb = load_word_embedding(opt.out_emb_path)
        model.out.weight.data.copy_(out_emb)

    if opt.norm_out_emb:
        model.out.weight.data =\
            model.norm_tensor(model.out.weight.data).detach()

    model.out.weight.requires_grad = opt.update_out_emb

    criterion = nn.CrossEntropyLoss()
    #  optimizer = optim.SGD(model.parameters(), lr=float(opt.lr))
    optimizer = optim.SGD(filter(
        lambda p: p.requires_grad,
        model.parameters()
    ), lr=float(opt.lr))

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
    for epoch in range(1, int(opt.epoch) + 1):
        start_time = time.time()
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        count = 0
        for batch_count, batch_train in enumerate(train_iter, 1):
            optimizer.zero_grad()
            text = batch_train.text.t().contiguous()
            target = batch_train.target.t().contiguous()
            for i in range(text.size(1) - opt.window_len + 1):
                text_ = text[:, i:i+opt.window_len]
                target_ = target[:, i+opt.window_len-1]
                prediction = model(text_)
                loss = criterion(
                    prediction, target_
                )
                loss.backward()
                #  print(f"{loss.item()}-------")
                #  print(f"text:{text_}")
                #  print(f"target:{target_}")

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

        if logger:
            logger.info('| epoch {:3d} | train_loss {:5.2f} '
                        '| val_ppl {:8.5f} | time {:5.1f}s'.format(
                            epoch,
                            train_loss,
                            val_ppl,
                            elapsed))

        # Saving model
        if epoch % opt.every_n_epoch_save == 0:
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
    #  test_loss = evaluation_similarity(test_iter)
    #  test_ppl = test_loss
    test_loss = evaluation(test_iter)
    test_ppl = math.exp(test_loss)
    if logger:
        logger.info("test_ppl: {:5.5f}".format(test_ppl))

    # saving output embeddings
    #  save_word_embedding(
        #  TEXT.vocab.itos,
        #  model.out.weight.data,
        #  f"{opt.window_len}mlplen_{opt.epoch}epoch_outemb.txt"
    #  )

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
