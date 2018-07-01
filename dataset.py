#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : Dataset class using torchtext
"""

import os
import torchtext
import spacy


def create_lm_dataset():
    """create language modeling dataset.
    :returns: iterators for train, test and valid dataset

    """
    # Using spacy to tokenize text
    spacy_en = spacy.load('en')
    def tokenize(text):
        return [item.text for item in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenize
    )

    wikitext_dir = os.path.expanduser("~/common_corpus/")
    train, valid, test = torchtext.datasets.WikiText2.splits(
        text_field=TEXT,
        root=wikitext_dir
    )

    vectors_dir = os.path.expanduser("~/common_corpus/")
    TEXT.build_vocab(train, vectors="glove.6B.50d", vectors_cache=vectors_dir)

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=5,
        bptt_len=3,
        device=-1,
        repeat=False
    )
    for batch in train_iter:
        print(batch.text)
        break
    return (train_iter, test_iter, val_iter)


if __name__ == "__main__":
    #  unit test
    create_lm_dataset()
