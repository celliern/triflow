#!/usr/bin/env python
# coding=utf8

import logging
import numpy as np

import datreant.core as dtr
from toolz import curry
from collections import deque

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


@curry
def run_simulation(simul, history_len=None):
    init = (simul.t, simul.fields.rec.T)
    history = deque(init, history_len)
    for fields, t in simul:
        simul.history.append((t, fields.rec.T))
        logger.info(f"id: {simul.id}-{simul.pars['model']}, "
                    f"time: {simul.t}")
    return simul.id, simul.pars, history


@curry
def save_result(simul_id, parameters, history, path_data='.'):
    logger.info("\tsaving id: %s" % (simul_id))
    treant = dtr.Treant(f"{path_data}/{simul_id}")
    for key, value in parameters.item():
        try:
            treant.categories[key] = value
        except ValueError:
            pass
    ts, fields = zip(*history)
    tosave = {'t': ts, 'fields': fields}
    np.savez(f"{treant.path}/data.npz", **tosave)
    np.save(f"{treant.path}/parameters", parameters)

    logger.info("\tid: %s saved" % (simul_id))
