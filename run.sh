#!/bin/sh

# ----------------------------------------------------------------------
# Author      : Yukun Feng
# Date        : 2018/07/31
# Email       : yukunfg@gmail.com
# Description : Running NNLM
# ----------------------------------------------------------------------

# python ./train.py  -device "cuda:0" -log ""
# python ./train.py  -device "cuda:0" -seed 2
# python ./train.py  -device "cuda:0"
# python ./train.py  -device "cuda:0" -seed 3
# python ./train.py  -device "cuda:0" -seed 13 -bptt_len 15 -batch_size 25

# python ./train.py  -device "cuda:0" -not_update_input_emb

# Generate out emb
python ./train.py  -device "cuda:0" -seed 19 -bptt_len 10 -batch_size 400 -update_out_emb -window_len 1 -word_dim 100 -out_emb_path "./wiki.train.tokens.100d.cbow.txt"
# python ./train.py  -device "cuda:0" -seed 19 -bptt_len 10 -batch_size 400 -update_out_emb -random_outemb -window_len 1 -word_dim 100

# python ./train.py  -device "cuda:0" -seed 190 -bptt_len 15 -batch_size 25 -out_emb_path "./3bptt_2epoch_outemb.txt" -update_out_emb
