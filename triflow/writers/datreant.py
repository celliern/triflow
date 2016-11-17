#!/usr/bin/env python
# coding=utf8

import logging

import datreant.core as dtr
import numpy as np
from path import path

from triflow.displays import full_display, simple_display
from triflow.misc import coroutine


def get_datreant_conf(simul):
    path_data = simul.conf.get('treant.path', '.')
    simul_name = simul.conf.get('treant.name', simul.id)
    compressed = simul.conf.get('treant.compressed', False)
    return path_data, simul_name, compressed


def datreant_init(path_data, simul_name, parameters):
    treant = dtr.Treant(path(path_data) / simul_name)
    for key in parameters:
        try:
            treant.categories[key] = parameters[key]
        except TypeError:
            treant.categories[key] = float(parameters[key])
    treant.categories['t'] = 0
    treant.categories['i'] = 0
    return treant


def datreant_save(treant, i, t, tosave, compressed=False):
    treant.categories['t'] = t
    treant.categories['i'] = i
    if not compressed:
        np.savez(path(treant.abspath) / 'data', **tosave)
    else:
        np.savez_compressed(path(treant.abspath) / 'data', **tosave)


@coroutine
def datreant_steps_writer(simul):
    path_data, simul_name, compressed = get_datreant_conf(simul)
    treant = datreant_init(path_data, simul_name, simul.pars)
    display = full_display(simul)

    while True:
        simul = yield
        t, field = display.send(simul)
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['t'] = t
        tosave['x'] = simul.x
        datreant_save(treant, simul.i, simul.t, tosave, compressed)


@coroutine
def datreant_step_writer(simul):
    path_data, simul_name, compressed = get_datreant_conf(simul)
    treant = datreant_init(path_data, simul_name, simul.pars)
    display = simple_display(simul)
    while True:
        simul = yield
        t, field = display.send(simul)
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['x'] = simul.x
        datreant_save(treant, simul.i, simul.t, tosave, compressed)


datreant_step_writer.writer_type = 'datreant'
datreant_steps_writer.writer_type = 'datreant'
