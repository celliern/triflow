#!/usr/bin/env python
# coding=utf8

from triflow.core.model import Model, load_model

import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
