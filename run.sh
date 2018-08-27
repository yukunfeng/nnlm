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
# python ./train.py  -device "cuda:0" -seed 9 -bptt_len 10 -batch_size 400 -update_out_emb -window_len 2 -word_dim 100 -out_emb_path "./wiki.train.tokens.100d.cbow.txt"
# python ./train.py  -device "cuda:0" -seed 19 -bptt_len 10 -batch_size 400 -update_out_emb -random_outemb -window_len 1 -word_dim 100

# python ./train.py  -device "cuda:0" -seed 190 -bptt_len 15 -batch_size 25 -out_emb_path "./3bptt_2epoch_outemb.txt" -update_out_emb


# Generate out emb and whole model saved
# python ./train.py  -device "cuda:0" -seed 90 -bptt_len 10 -batch_size 800 -update_out_emb \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.numupper.freq3.preprocessed.850d.cbow.txt" \
# -window_len 1 -random_outemb 

# Generate out emb and whole model saved for ptb
# python ./train.py  -device "cuda:0" -seed 1000 -bptt_len 10 -batch_size 800 -update_out_emb -data_type "ptb" \
# -input_vector "~/common_corpus/penn-treebank/ptb.train.txt.cbow.850d" \
# -window_len 1 -random_outemb 

# python ./train.py  -device "cuda:0" -seed 9089 -bptt_len 10 -batch_size 800 -update_out_emb \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.325d.cbow.txt" \
# -window_len 2 -random_outemb 
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.100d.cbow.lower.txt" \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.100d.cbow.txt" \

# python ./train.py  -device "cuda:0" -seed 9099 -bptt_len 10 -batch_size 800 -update_out_emb -window_len 2 -word_dim 100 -random_outemb -every_n_epoch_save 7 -epoch 7
# python ./train.py  -device "cuda:0" -seed 99 -bptt_len 10 -batch_size 200 -update_out_emb -window_len 2 -word_dim 100 -out_emb_path "../common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.200d.cbow.txt" -every_n_epoch_save 8


###########################
#  full vocab experiment  #
###########################

python ./train.py  -device "cuda:0" -seed 200 -bptt_len 10 -batch_size 15000 -update_out_emb -data_type "wiki2" \
-input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.cbow.8.5kvocab.200d" \
-window_len 1 -random_outemb  -lr 1 -decay 0.5
