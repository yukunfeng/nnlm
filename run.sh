#!/bin/sh

# ----------------------------------------------------------------------
# Author      : Yukun Feng
# Date        : 2018/07/31
# Email       : yukunfg@gmail.com
# Description : Running NNLM
# ----------------------------------------------------------------------

# python ./train.py  -device "cuda:0" -log ""
# python ./train.py  -device "cuda:0" -seed 2
python ./train.py  -device "cuda:1"
# python ./train.py  -device "cuda:0" -not_update_input_emb
