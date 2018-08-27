#!/usr/bin/env python3

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


def main():
    pass

if __name__ == "__main__":
    main()
