#!/usr/bin/env python3
"""
Author      : Yukun Feng
Date        : 2018/08/27
Email       : yukunfg@gmail.com
Description : Train word vectors based on co-matrix
"""

from collections import Counter
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
        self.matrix_make()

    def matrix_make(self):
        matrix = {}
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
                        continue
                    if word not in matrix:
                        matrix[word] = {}
                    if words[count + 1] not in matrix[word]:
                        matrix[word][words[count + 1]] = 0
                    matrix[word][words[count + 1]] += 1
        self.matrix = matrix

    def print(self, num=10):
        words = self.matrix.keys()
        #  words = [self.vocab.itos[word] for word in words]
        for count, word in enumerate(words, 1):
            if count == num:
                break

            out_line = f"{self.vocab.itos[word]}: "
            context_words = self.matrix[word]
            col_count = 0
            for context_word, freq in context_words.items():
                context_word_str = self.vocab.itos[context_word]
                col_count += 1
                out_line += f"({context_word_str} {freq} )"
                if col_count == num:
                    break
            print(out_line) 


def main():
    matrix = WordMatrix("./test.txt")
    matrix.print()

if __name__ == "__main__":
    main()
