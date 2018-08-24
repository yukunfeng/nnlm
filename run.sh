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
# python ./train.py  -device "cuda:0" -seed 111 -bptt_len 4 -batch_size 25 -update_out_emb -random_outemb
# python ./train.py  -device "cuda:0" -seed 211 -bptt_len 5 -batch_size 20 -update_out_emb -random_outemb

# pretraining rnn
# python ./train.py  -device "cuda:0" -seed 100 -bptt_len 2 -batch_size 100 -update_out_emb -out_emb_path "../nnlm/2mlplen_8epoch_outemb.txt" -every_n_epoch_decay 2 -epoch 30

# python ./train.py  -device "cuda:0" -seed 231 -bptt_len 35 -batch_size 20 -out_emb_path "../nnlm/2mlplen_8epoch_outemb.txt" -update_out_emb -epoch 40
# python ./train.py  -device "cuda:0" -seed 500 -bptt_len 35 -batch_size 20 -out_emb_path "./35bptt_40epoch_outemb1st.txt" -update_out_emb -epoch 40
# python ./train.py  -device "cuda:0" -seed 131 -bptt_len 35 -batch_size 20 -out_emb_path "/home/lr/yukun/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.200d.cbow.txt" -update_out_emb -epoch 20
# python ./train.py  -device "cuda:0" -seed 131 -bptt_len 35 -batch_size 20 -update_out_emb -epoch 20 -random_outemb

# new opt including input vector
# python ./train.py  -device "cuda:0" -seed 0 -bptt_len 10 -batch_size 20 -update_out_emb \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.100d.cbow.txt" \
# -out_emb_path "~/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.100d.cbow.txt"
# python ./train.py  -device "cuda:0" -seed 0 -bptt_len 10 -batch_size 20 -update_out_emb \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.100d.cbow.lower.txt" \
# -out_emb_path "~/common_corpus/wikitext-2/wikitext-2/wiki.train.100d.cbow.lower.txt"

# big training
# python ./train.py  -device "cuda:1" -seed 104 -bptt_len 35 -batch_size 20 -update_out_emb -epoch 40 \
# -input_vector "~/common_corpus/2mlplen_8epoch_650d_outemb.txt" \
# -out_emb_path "~/common_corpus/2mlplen_8epoch_650d_outemb.txt"

# big training
# python ./train.py  -device "cuda:1" -seed 1000 -bptt_len 20 -batch_size 20 -update_out_emb -epoch 25 \
# -input_vector "~/common_corpus/2mlplen_8epoch_650d_outemb.txt" \
# -out_emb_path "~/common_corpus/2mlplen_8epoch_650d_outemb.txt"
# python ./train.py  -device "cuda:1" -seed 1050 -bptt_len 45 -batch_size 20 -update_out_emb -epoch 25  \
# -input_vector "~/common_corpus/2mlplen_8epoch_650d_outemb.txt" \
# -out_emb_path "~/common_corpus/2mlplen_8epoch_650d_outemb.txt"

# python ./train.py  -device "cuda:1" -seed 1050 -bptt_len 65 -batch_size 20 -update_out_emb -epoch 25 \
# -input_vector "~/common_corpus/2mlplen_8epoch_650d_outemb.txt" \
# -out_emb_path "~/common_corpus/2mlplen_8epoch_650d_outemb.txt"
# tied
# python ./train.py -tied  -device "cuda:1" -seed 205 -bptt_len 80 -batch_size 20 -update_out_emb -epoch 40 \
# -input_vector "~/common_corpus/2mlplen_8epoch_650d_outemb.txt" \
# -out_emb_path "~/common_corpus/2mlplen_8epoch_650d_outemb.txt"

# using generated out emb by rnn
# python ./train.py -tied  -device "cuda:1" -seed 2 -bptt_len 80 -batch_size 20 -epoch 40 \
# -input_vector "~/common_corpus/1mlplen_8epoch_10529v.850d.outemb.txt" \
# -out_emb_path "~/common_corpus/1mlplen_8epoch_10529v.850d.outemb.txt"

# using cbow 200 may intrig gradient explode
emb="~/common_corpus/wikitext-103/wikitext-103/wiki.train.tokens.preprocessed.cbow.850d"
python ./train.py -tied  -device "cuda:0" -seed 1 -bptt_len 80 -batch_size 20 -epoch 40 \
-input_vector $emb \
-out_emb_path $emb -every_n_epoch_save 8 -data_type "wiki3"

# ptb for random
# emb="~/common_corpus/1mlplen_8epoch_outemb.ptb.850d.txt"
# emb="~/common_corpus/1mlplen_8epoch_outemb.100d.ptb.txt"
# python ./train.py -update_out_emb  -device "cuda:0" -seed 240 -bptt_len 22 -batch_size 1 -epoch 30 -data_type "ptb" \
# -input_vector $emb -tied -out_emb_path $emb

# for random
# python ./train.py -random_outemb -update_out_emb  -device "cuda:0" -seed 1 -bptt_len 80 -batch_size 20 -epoch 40 \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.numupper.freq3.preprocessed.850d.cbow.txt" \

# python ./train.py -tied  -device "cuda:0" -seed 1 -bptt_len 80 -batch_size 20 -epoch 20 \
# -input_vector "~/common_corpus/wikitext-2/wikitext-2/wiki.train.numupper.freq3.preprocessed.850d.cbow.txt" \
# -out_emb_path "~/common_corpus/wikitext-2/wikitext-2/wiki.train.numupper.freq3.preprocessed.850d.cbow.txt"
