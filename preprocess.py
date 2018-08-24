#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/06/20
Email       : yukunfg@gmail.com
Description : Preprocess text
"""

from collections import Counter
import argparse


def preprocess(args):
    """preprocess function"""

    # At first do counting
    total_counter = Counter()
    with open(args.file_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines
            if line == "":
                continue
            total_counter.update(line.split())

    # Second do preprocessing
    with open(args.file_path, 'r') as fh:
        for line in fh:
            tokens = line.split()
            for count, token in enumerate(tokens, 0):
                freq = total_counter[token]
                if freq <= args.replace_freq:
                    tokens[count] = args.unk_symbol
                    continue
                if token[0].isupper():
                    tokens[count] = args.unk_symbol
                if token.isnumeric():
                    tokens[count] = args.unk_symbol
                    continue
            new_line = " ".join(tokens)
            print(new_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess text and output to stdout',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-replace_freq', default=3, type=int,
        help='replace words to unks whose freq <= 3'
    )
    parser.add_argument(
        '-unk_symbol', default='<unk>', type=str,
        help='unk symbol to be replaced'
    )
    parser.add_argument(
        '-file_path', type=str, required=True,
        help='file_path'
    )
    args = parser.parse_args()
    preprocess(args)
