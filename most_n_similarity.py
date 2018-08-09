#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/08/09
Email       : yukunfg@gmail.com
Description : Extract most N similar words
"""

import string
from os.path import expanduser
from collections import Counter
import word2vec


def corpus_freq(file_path):
    token_counter = Counter()
    with open(file_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines
            if line == "":
                continue
            line = [i for i in line if i not in string.punctuation]
            line = "".join(line)
            tokens = line.split()
            line_counter = Counter(tokens)
            token_counter.update(line_counter)
    return token_counter


def main():
    model = word2vec.load(
        "./wiki.train.tokens.100d.cbow.txt",
        kind="txt"
    )
    corpus_path = "~/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens"
    corpus_path = expanduser(corpus_path)
    token_counter = corpus_freq(corpus_path)
    threshold = 0.5

    for i in range(0, model.vocab.shape[0]):
        word = model.vocab[i]
        indexes, metrics = model.cosine(word, model.vocab.shape[0])
        for count, metric in enumerate(metrics, 1):
            if metric <= threshold:
                break
        print(f"{word} {count + 1} {token_counter[word]}")

if __name__ == "__main__":
    main()
    #  position_less_than_n_test()
