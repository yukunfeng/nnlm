#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Dataset class using torchtext
"""

import os
import argparse
import opts
import torchtext
import spacy
from spacy.symbols import ORTH
from utils.utils import *


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    return opt


def create_lm_dataset_for_ptb(opt, logger=None):
    # Using spacy to tokenize text
    spacy_en = spacy.load('en')
    # Add <unk> special case is due to wiki text which has raw <unk>
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def tokenize(text):
        """tokenize sentence"""
        return [item.text for item in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(
        sequential=True,
        #  tokenize=tokenize,
        use_vocab=True,
        include_lengths=True,
        init_token='<s>',
        eos_token='</s>'
    )
    fields = [("text", TEXT)]

    resources_dir = os.path.expanduser(opt.resources_dir)
    resources_dir = f"{resources_dir}/penn-treebank/"
    train, valid, test = torchtext.data.TabularDataset.splits(
        fields=fields,
        path=resources_dir,
        train="ptb.train.txt",
        validation="ptb.valid.txt",
        test="ptb.test.txt",
        format="csv"
    )
    if logger:
        logger.info(f"train token: {len(train.examples[0].text)}")
        logger.info(f"test token: {len(test.examples[0].text)}")
        logger.info(f"valid token: {len(valid.examples[0].text)}")

    device = torch.device(opt.device)
    if opt.input_vector is not None:
        opt.input_vector = os.path.expanduser(opt.input_vector)
        head, tail = os.path.split(opt.input_vector)
        torchtext_vectors = torchtext.vocab.Vectors(name=tail, cache=head)
        torchtext_vectors.vectors.to(device)
        #  print(f"len: {len(torchtext_vectors.stoi)}")
        #  print(f"size: {torchtext_vectors.vectors.size()}")
        # Here the list of list is to simulate the real dataset
        # where first dim is sentence and second is word.
        limited_train = [[word] for word in torchtext_vectors.stoi.keys()]
        TEXT.build_vocab(limited_train, vectors=torchtext_vectors)
    else:
        TEXT.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, valid, test),
        batch_sizes=(opt.batch_size, opt.batch_size, opt.batch_size),
        device=opt.device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        repeat=False
    )
    return (TEXT, train_iter, test_iter, val_iter)


def create_lm_dataset(opt, logger=None):
    """create language modeling dataset.
    :returns: iterators for train, test and valid dataset

    """
    # Using spacy to tokenize text
    spacy_en = spacy.load('en')
    # Add <unk> special case is due to wiki text which has raw <unk>
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def tokenize(text):
        """tokenize sentence"""
        return [item.text for item in spacy_en.tokenizer(text)]

    is_lower = True
    if opt.data_type == "ptb":
        is_lower = False
    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        lower=is_lower
    )

    resources_dir = os.path.expanduser(opt.resources_dir)
    if opt.data_type == "wiki3":
        train, valid, test = torchtext.datasets.WikiText103.splits(
            text_field=TEXT,
            root=resources_dir
        )
    if opt.data_type == "wiki2":
        train, valid, test = torchtext.datasets.WikiText2.splits(
            text_field=TEXT,
            root=resources_dir
        )
    if opt.data_type == "ptb":
        train, valid, test = torchtext.datasets.PennTreebank.splits(
            text_field=TEXT,
            root=resources_dir
        )

    if logger:
        logger.info(f"train token: {len(train.examples[0].text)}")
        logger.info(f"test token: {len(test.examples[0].text)}")
        logger.info(f"valid token: {len(valid.examples[0].text)}")

    device = torch.device(opt.device)
    if opt.input_vector is not None:
        opt.input_vector = os.path.expanduser(opt.input_vector)
        head, tail = os.path.split(opt.input_vector)
        torchtext_vectors = torchtext.vocab.Vectors(name=tail, cache=head)
        torchtext_vectors.vectors.to(device)
        #  print(f"len: {len(torchtext_vectors.stoi)}")
        #  print(f"size: {torchtext_vectors.vectors.size()}")
        # Here the list of list is to simulate the real dataset
        # where first dim is sentence and second is word.
        limited_train = [[word] for word in torchtext_vectors.stoi.keys()]
        TEXT.build_vocab(limited_train, vectors=torchtext_vectors)
    else:
        TEXT.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=opt.batch_size,
        bptt_len=opt.bptt_len,
        device=device,
        repeat=False
    )
    return (TEXT, train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    opt = parse_args()
    #  logger = get_logger(opt.log_file)
    logger = None
    #  TEXT, train_iter, test_iter, val_iter = create_lm_dataset(
        #  opt, logger=logger
    #  )
    TEXT, train_iter, test_iter, val_iter = create_lm_dataset_for_ptb(
        opt, logger=logger
    )
    #  print(f"{TEXT.vocab.vectors.size()}")
    #  print(f"{len(TEXT.vocab.itos)}")

    for batch_count, batch_data in enumerate(val_iter, 1):
        if opt.data_type == "ptb":
            text, lengths = batch_data.text
            target = text[1:, :]
            text = text[0:-1, :]
        else:
            text = batch_data.text
            target = batch_data.target
        print("---------")
        print("text words")
        print(word_ids_to_sentence(text, TEXT.vocab, word_len=12))
        print("target words")
        print(word_ids_to_sentence(target, TEXT.vocab, word_len=12))
        print("text number")
        print(text)
        print("target number")
        print(target)
        print("length")
        print(lengths)
        if batch_count == 10:
            break
