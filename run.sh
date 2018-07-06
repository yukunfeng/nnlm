#!/bin/sh

python train.py  -rnn_type RNN -bidirectional 1 -num_layers 1 -vocab_size 50 -word_dim 30 -dropout 0
