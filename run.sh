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
python ./train.py  -device "cuda:0" -seed 211 -bptt_len 5 -batch_size 20 -update_out_emb -random_outemb

# python ./train.py  -device "cuda:0" -seed 231 -bptt_len 35 -batch_size 20 -out_emb_path "../nnlm/2mlplen_8epoch_outemb.txt" -update_out_emb -epoch 40
# python ./train.py  -device "cuda:0" -seed 500 -bptt_len 35 -batch_size 20 -out_emb_path "./35bptt_40epoch_outemb1st.txt" -update_out_emb -epoch 40
# python ./train.py  -device "cuda:0" -seed 131 -bptt_len 35 -batch_size 20 -out_emb_path "/home/lr/yukun/common_corpus/wikitext-2/wikitext-2/wiki.train.tokens.200d.cbow.txt" -update_out_emb -epoch 20
# python ./train.py  -device "cuda:0" -seed 131 -bptt_len 35 -batch_size 20 -update_out_emb -epoch 20 -random_outemb
