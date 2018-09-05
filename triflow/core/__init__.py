#!/usr/bin/env python
# coding=utf-8

import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
