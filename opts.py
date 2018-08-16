#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : options from opennmt-py
"""


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Device
    group = parser.add_argument_group('Device')
    group.add_argument(
        '-device',
        default="cuda:0",
        help="e.g., cpu or cuda:1"
    )

    # Data options
    group = parser.add_argument_group('Data')

    group.add_argument('-resources_dir',
                       default='~/common_corpus/',
                       help="where wiki.. glove are")
    group.add_argument('-vector_type',
                       default="glove.6B.100d",
                       help="E.g., glove.6b.300d")
    group.add_argument('-batch_size', type=int,
                       default=20,
                       help="batch size")
    group.add_argument('-bptt_len', type=int,
                       default=10,
                       help="bptt length")

    # Model options
    group = parser.add_argument_group('Model')
    group.add_argument(
        '-update_inputemb',
        default=True,
        action='store_true',
        help="whether train inputembeddings 1 to train 0 to not train"
    )
    group.add_argument(
        '-random_outemb',
        default=False,
        action='store_true',
        help="whether randomly init outemb"
    )
    group.add_argument('-out_emb_path',
                       default='./10bptt_8epoch_outemb.txt',
                       help="out emb path")
    group.add_argument(
        '-update_out_emb',
        default=False,
        action='store_true',
        help="whether train input embeddings 1 to train 0 to not train"
    )
    group.add_argument(
        '-norm_out_emb',
        default=False,
        action='store_true',
        help="whether norm out emb"
    )
    group.add_argument('-save', default="mlp.model", help="the saving path")
    group.add_argument(
        '-every_n_epoch_save',
        default=40,
        type=int,
        help="every this epoch saving model"
    )
    group.add_argument('-seed', default=0, help="random seed", type=int)
    group.add_argument(
        '-tied', default=False,
        help="tied input and output embedding",
        action='store_true'
    )
    group.add_argument(
        '-window_len', default=5,
        help="window_len",
        type=int
    )
    group.add_argument(
        '-word_dim', default=50,
        help="window_len",
        type=int
    )
    group.add_argument('-epoch', default=8, help="epoch", type=int)
    group.add_argument('-lr', default=0.5, help="learning rate", type=float)
    group.add_argument('-rnn_type', default='GRU', help="type", type=str)
    group.add_argument('-bidirectional', default=False, type=bool)
    group.add_argument(
        '-num_layers',
        default=1,
        type=int,
        help="number of layers"
    )

    group.add_argument(
        '-dropout',
        default=0.0,
        help="dropout",
        type=float
    )

    group = parser.add_argument_group('Logging')
    group.add_argument('-log_file', type=str, default="log",
                       help="Output logs to a file under this path.")
