#!/usr/bin/env python3

"""
Author      : Yukun Feng
Date        : 2018/07/01
Email       : yukunfg@gmail.com
Description : options from opennmt-py
"""


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')

    group.add_argument('-resources_dir', required=True,
                       default='',
                       help="where wiki.. glove are")
    group.add_argument('-vector_type', required=True,
                       default="glove.6b",
                       help="E.g., glove.6b")
    group.add_argument('-batch_size', required=True, type=int,
                       default=64,
                       help="batch size")
    group.add_argument('-bptt_len', required=True, type=int,
                       default=10,
                       help="bptt length")

    # Model options
    group = parser.add_argument_group('Model')
    group.add_argument('-rnn_type', default='rnn', help="type")
    group.add_argument('-bidirectional', default=False, type=bool)
    group.add_argument(
        '-num_layers',
        default=1,
        type=int,
        help="number of layers"
    )
    group.add_argument(
        '-vocab_size',
        required=True,
        help="vocab_size",
        type=int
    )
    group.add_argument(
        '-word_dim',
        required=True,
        help="word_dim",
        type=int
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
