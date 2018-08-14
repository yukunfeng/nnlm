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


def create_lm_dataset(resources_dir, vector_type, batch_size, bptt_len, device, logger=None):
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

    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize,
        lower=True
    )

    wikitext_dir = os.path.expanduser(resources_dir)
    train, valid, test = torchtext.datasets.WikiText2.splits(
        text_field=TEXT,
        root=wikitext_dir
    )

    if logger:
        logger.info(f"train token: {len(train.examples[0].text)}")
        logger.info(f"test token: {len(test.examples[0].text)}")
        logger.info(f"valid token: {len(valid.examples[0].text)}")

    vectors_dir = os.path.expanduser(resources_dir)

    vectors_path = "{}/{}.txt".format(vectors_dir, vector_type)
    vocab_from_vectors = []
    with open(vectors_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines
            if line == "":
                continue
            word = [line[0:line.find(" ")]]
            vocab_from_vectors.append(word)

    TEXT.build_vocab(
        vocab_from_vectors,
        vectors=vector_type,
        vectors_cache=vectors_dir
    )

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=bptt_len,
        device=device,
        repeat=False
    )
    return (TEXT, train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    opt = parse_args()
    logger = get_logger(opt.log_file)
    TEXT, train_iter, test_iter, val_iter = create_lm_dataset(
        resources_dir=opt.resources_dir,
        vector_type=opt.vector_type,
        batch_size=opt.batch_size,
        bptt_len=opt.bptt_len,
        device=opt.device,
        logger=logger
    )

    for batch_count, batch_data in enumerate(train_iter, 1):
        text = batch_data.text.t().contiguous()
        target = batch_data.target.t().contiguous()
        strings = word_ids_to_sentence(
            text[:, 0:10], TEXT.vocab,
            word_len=12
        )
        print(strings)
        break
