#!/usr/bin/env python
# coding=utf8

import functools as ft
import logging

import datreant.core as dtr
import numpy as np
from path import path
from triflow.plugins.displays import simple_display


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


def datreant_save(path_data, simul_name, i, t,
                  tosave, compressed, lock):
    treant = dtr.Treant(path(path_data) / simul_name)
    logging.debug('save path: %s ' % treant.abspath)
    treant.categories['t'] = t
    treant.categories['i'] = i
    with lock:
        if not compressed:
            np.savez(path(treant.abspath) / 'data', **tosave)
        else:
            np.savez_compressed(path(treant.abspath) / 'data', **tosave)


def datreant_append(path_data, simul_name, i, t,
                    tosave, compressed, lock):
    treant = dtr.Treant(path(path_data) / simul_name)
    logging.debug('save path: %s ' % treant.abspath)
    treant.categories['t'] = t
    treant.categories['i'] = i
    with lock:
        try:
            old_data = np.load(path(treant.abspath) / 'data.npz')
            for fieldname in old_data.iterkeys():
                oldies = old_data[fieldname]
                tosave[fieldname] = np.insert(oldies, 0,
                                              tosave[fieldname], axis=0)
                logging.debug(tosave[fieldname].shape)
        except FileNotFoundError:
            tosave = {key: value[np.newaxis, :]
                      for (key, value)
                      in tosave.items()}
        finally:
            if not compressed:
                np.savez(path(treant.abspath) / 'data', **tosave)
            else:
                np.savez_compressed(path(treant.abspath) / 'data', **tosave)


def datreant_step_writer(simul):
    path_data, simul_name, compressed = get_datreant_conf(simul)
    datreant_init(path_data, simul_name, simul.pars)
    display = simple_display(simul)
    for t, field in display:
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['x'] = simul.x
        simul.simuloop.run_in_executor(None,
                                       ft.partial(
                                           datreant_save,
                                           path_data,
                                           simul_name,
                                           simul.i,
                                           t,
                                           tosave,
                                           compressed,
                                           simul.datalock))
        yield


def datreant_steps_writer(simul):
    path_data, simul_name, compressed = get_datreant_conf(simul)
    datreant_init(path_data, simul_name, simul.pars)
    display = simple_display(simul)
    for t, field in display:
        tosave = {name: field[name]
                  for name
                  in simul.solver.fields}
        tosave['t'] = np.array([t])
        tosave['x'] = simul.x
        simul.simuloop.run_in_executor(None,
                                       ft.partial(
                                           datreant_append,
                                           path_data,
                                           simul_name,
                                           simul.i,
                                           t,
                                           tosave,
                                           compressed,
                                           simul.datalock))
        yield


datreant_step_writer.writer_type = 'datreant'
datreant_steps_writer.writer_type = 'datreant'
