#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
from project_path import *
from triflow.core.solver import Solver

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def load_data(sample_id):
    """
        load simulation field result (not metadata),
        in a lazy way (see numpy npz file format)
    """
    logging.info('reading %s' % sample_id)
    data = np.load(data_dir / 'runs' / sample_id / 'data.npz')
    return data
