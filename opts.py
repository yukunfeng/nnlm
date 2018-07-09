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
    group.add_argument('-device',
                       default="cpu",
                       help="e.g., cpu or cuda:1")

    # Data options
    group = parser.add_argument_group('Data')

    group.add_argument('-resources_dir',
                       default='~/common_corpus/',
                       help="where wiki.. glove are")
    group.add_argument('-vector_type',
                       default="glove.6B.50d",
                       help="E.g., glove.6b.300d")
    group.add_argument('-batch_size', type=int,
                       default=5,
                       help="batch size")
    group.add_argument('-bptt_len', type=int,
                       default=10,
                       help="bptt length")

    # Model options
    group = parser.add_argument_group('Model')
    group.add_argument(
        '-input_embeddings_trainable',
        default=True,
        type=bool,
        help="whether train inputembeddings"
    )
    group.add_argument('-epoch', default=30, help="epoch")
    group.add_argument('-lr', default=0.01, help="learning rate")
    group.add_argument('-rnn_type', default='RNN', help="type")
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
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")
    group.add_argument('-log_file', type=str, default="",
                       help="Output logs to a file under this path.")
