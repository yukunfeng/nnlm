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
from spacy.symbols import ORTH


def create_lm_dataset(resources_dir, vector_type, batch_size, bptt_len, device):
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
    create_lm_dataset(
        resources_dir="",
        vector_type="",
        batch_size="",
        bptt_len="",
        device=""
    )
