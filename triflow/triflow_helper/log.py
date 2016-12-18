#!/usr/bin/env python
# coding=utf8

import logging

from path import Path


def init_log(log_dir, log_name, level_stream, level_file):
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    logger = logging.getLogger()
    logger.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level_stream)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / '%s.log' % log_name)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.setLevel(level_file)
