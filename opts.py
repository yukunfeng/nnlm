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

    group.add_argument('-train', required=True,
                       help="Path to the training data")
    group.add_argument('-test', required=True,
                       help="Path to the test data")
    group.add_argument('-valid', required=True,
                       help="Path to the validation data")

    group = parser.add_argument_group('Logging')
    group.add_argument('-report_every', type=int, default=100000,
                       help="Report status every this many sentences")
    group.add_argument('-log_file', type=str, default="",
                       help="Output logs to a file under this path.")
