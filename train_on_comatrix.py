#!/usr/bin/env python3
"""
Author      : Yukun Feng
Date        : 2018/08/27
Email       : yukunfg@gmail.com
Description : Train word vectors based on co-matrix
"""

import argparse
from utils.utils import *


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-file', help='file path', required=True
    )
    parser.add_argument(
        '-log', default='log.matrix', help='log file'
    )
    parser.add_argument('-w', action='store_true')
    opt = parser.parse_args()
    return opt


def matrix_make(opt):
    "build matrix"
    matrix = {}
    with open(opt.file_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines
            if line == "":
                continue
            words = line.split()
            for count, word in enumerate(words, 0):
                if word not in matrix:
                    matrix[word] = {}
                if count < len(words) - 1 \
                   and words[count + 1] not in matrix[word]:
                    matrix[word][words[count + 1]] = 1
                matrix[word][words[count + 1]] += 1
    return matrix


def print_matrix(matrix):
    words = matrix.keys()
    for word in words:
        out_line = f"{word}: "
        context_words = matrix[word]
        for context_word, freq in context_words:
            out_line += f"({context_word} {freq} )"
        print(out_line) 


def main(opt, logger=None):
    matrix = matrix_make(opt)
    print_matrix(matrix)


if __name__ == "__main__":
    opt = parse_args()
    logger = get_logger(opt.log_file)
