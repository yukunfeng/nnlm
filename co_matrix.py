#!/usr/bin/env python3
"""
Author      : Yukun Feng
Date        : 2018/08/27
Email       : yukunfg@gmail.com
Description : Train word vectors based on co-matrix
"""

from collections import Counter
import torch
import torchtext
from utils.utils import *


class WordMatrix(object):
    """word matrix"""
    def __init__(self, file_path):
        self.file_path = file_path 
        counter = Counter()
        with open(file_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                # Skip empty lines
                if line == "":
                    continue
                tokens = line.split()
                counter.update(tokens)
        self.vocab = torchtext.vocab.Vocab(counter, specials=[]) 
        self.vocab_size = len(self.vocab.itos)
        self.matrix_make()

    def matrix_make(self):
        matrix = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.int32)
        matrix.requires_grad = False
        with open(self.file_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                # Skip empty lines
                if line == "":
                    continue
                words = line.split()
                words = [self.vocab.stoi[word] for word in words]
                for count, word in enumerate(words, 0):
                    if count == len(words) - 1:
                        break
                    context_word = words[count + 1]
                    matrix[word][context_word] += 1
        self.matrix = matrix

    def print(self, num=10):
        for word in range(self.vocab_size):
            if word + 1 == num:
                break

            out_line = f"{self.vocab.itos[word]}: "
            freqs = self.matrix[word]
            for freq_count, freq in enumerate(freqs, 1):
                if freq_count == num:
                    break
                context_word_str = self.vocab.itos[freq_count - 1]
                out_line += f"({context_word_str} {freq} )"
            print(out_line) 


def main():
    matrix = WordMatrix("./test.txt")
    matrix.print()

if __name__ == "__main__":
    main()
