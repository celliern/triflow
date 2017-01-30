#!/usr/bin/env python
# coding=utf8

import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Model:
    """docstring for Model"""
    def __init__(self,
                 func: str or list or tuple,
                 vars: str or list or tuple):
        super(Model, self).__init__()
        self.arg = arg

