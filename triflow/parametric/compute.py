#!/usr/bin/env python
# coding=utf8

import logging
import pickle
from collections import ChainMap
from timeit import default_timer as timer

import datreant.core as dtr
import numpy as np
from scipy.signal import hamming
from toolz import curry
from triflow.core.solver import Solver
from triflow.plugins import displays, writers

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def init_field(field, parameters):
    endpoint = False if parameters['kind'] == 'per' else True
    x, dx = np.linspace(0, parameters['L'], parameters['Nx'],
                        endpoint=endpoint, retstep=True)
    if field == 'h':
        if parameters.get('hydro_jump', False):
            return ((-(np.tanh(
                (x - .1 * parameters['L']) / 20) + 1) / 2 + 1) * 1 + 1) / 2
        if parameters['kind'] == 'per':
            return hamming(parameters['Nx']) * .1 + .95
        return x * 0 + 1

    if field == 'q':
        if parameters.get('hydro_jump', False):
            return (((-(np.tanh(
                (x - .1 * parameters['L']) / 20) + 1) /
                2 + 1) * 1 + 1) / 2) ** 3 / 3
        if parameters['kind'] == 'per':
            return (hamming(parameters['Nx']) * .1 + .95) ** 3 / 3
        return x * 0 + 1 / 3

    if field == 'theta':
        if parameters.get('thermo_eq', False):
            return x * 0 + parameters['theta_flat']
        if parameters.get('thermo_hot', False):
            return x * 0 + 1
        if parameters.get('thermo_cold', False):
            return x * 0
        return x * 0

    if field == 'phi':
        if parameters.get('thermo_eq', False):
            return x * 0 + parameters['phi_flat']
        if parameters.get('thermo_hot', False):
            return - parameters['B'] * (x * 0 + 1)
        if parameters.get('thermo_cold', False):
            return - parameters['B'] * (x * 0)
        return x * 0

    if field == 's':
        return x * 0

    if 'T' in field:
        if parameters.get('thermo_eq', False):
            i = int(field[1:])
            Ny = int(parameters['model'].split('ff')[0])
            y = ((-np.cos(np.pi * i / (Ny - 1))) + 1) / 2
            return ((parameters['theta_flat'] - 1) *
                    y + 1 + x * 0)
        if parameters.get('thermo_hot', False):
            return x * 0 + 1
        if parameters.get('thermo_cold', False):
            return x * 0
        return x * 0


@curry
def init_simulation(additional_parameters, sample,
                    **kwargs):
    parameters = ChainMap(dict(sample), *additional_parameters)
    logger.debug('full parameters keys are: %s'
                 % ' - '.join(parameters.keys()))
    parameters['kind'] = sample['model'].split('_')[-1]
    for key, value in parameters.items():
        if callable(value):
            parameters[key] = value(**parameters)
    solver = Solver(parameters['model'])
    initial_fields = []
    for field in solver.fields:
        initial_fields.append(kwargs.get(field, init_field(field,
                                                           parameters)))
        if isinstance(parameters['%sini' % field], str):
            parameters['%sini' % field] = parameters[parameters['%sini' %
                                                                field]]
    simul = solver.start_simulation(initial_fields, 0,
                                    id=parameters['name'], **parameters)
    simul.set_display(kwargs.get('display', displays.full_display))
    simul.set_scheme(kwargs.get('scheme', 'ROS_vart'))
    for driver in kwargs.get('drivers', []):
        simul.add_driver(driver)

    signal = kwargs.get('signal', None)
    if callable(signal):
        simul.add_signal(*signal(**parameters))
    if isinstance(signal, list):
        simul.set_signal(*signal)
    simul.filter = lambda simul, U: simul.i % parameters.get('n_save', 1) == 0

    return simul


@curry
def run_simulation(simul, retry=0, down_factor=2):
    def run(simul):
        old_time = timer()
        for iteration in simul:
            simul.history = iteration
            new_time = timer()
            logger.info("\tid: %s-%s, time: %i in %.2f sec" % (simul.id,
                                                               simul
                                                               .pars['model'],
                                                               simul.t,
                                                               new_time -
                                                               old_time))
            old_time = new_time
        return iteration

    backup_simul = simul
    simul = backup_simul.copy()

    for i in range(retry + 1):
        try:
            logger.debug('trying to run simul %s' % simul.id)
            run(simul)
            logger.info("\tid: %s over" % (simul.id))
            break
        except RuntimeError:
            logger.warning('something wrong in the solver,'
                           'restarting with more conservative tol\n'
                           'attempt number %i' % i)
            backup_simul.pars['tol'] /= down_factor
            simul = backup_simul.copy()
        except MemoryError:
            logger.warning('MemoryError,'
                           'stopping solv, trying to solve state')
            simul.status = 'error'
            break
        else:
            logger.error("attempt number %i didn't succeed" % i)
            simul.status = 'error'
            break
    return simul


@curry
def save_result(path_data, simul, bin=False):
    logger.info("\tsaving id: %s" %
                (simul.id))
    writers.datreant.datreant_init(path_data, simul.id, simul.pars)
    ts, fields = simul.history
    tosave = {name: fields[name]
              for name
              in simul.solver.fields}
    tosave.update({name: value
                   for name, value
                   in simul.solver.compute_Hs(simul.U,
                                              **simul.pars).items()})
    tosave['t'] = ts
    tosave['x'] = simul.x
    writers.datreant.datreant_save(path_data, simul.id, simul.i, simul.t,
                                   tosave, False)
    tr = dtr.Treant(path_data / simul.id)
    tr.tags.add(simul.status)
    logger.info("\tid: %s saved" % (simul.id))
    if bin:
        with open(path_data / simul.id / 'simul.bin', 'wb') as f:
            pickle.dump(simul, f)
