#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
import toolz

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


@toolz.curry
def load_data(path, sample_id):
    """
        load simulation field result (not metadata),
        in a lazy way (see numpy npz file format)
    """
    logging.info('reading %s' % sample_id)
    data = np.load(path / sample_id / 'data.npz')
    return data


def compute_Nu(data, parameters, phi_flat=None):
    """
        load simulation field result (not metadata),
        in a lazy way (see numpy npz file format)
    """
    if phi_flat is None:
        return data['phi'] / parameters['phi_flat']
    return data['phi'] / phi_flat

def update_field(helpers, data, parameters):
    data = dict(data)
    data.update({})
