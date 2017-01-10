#!/usr/bin/env python
# coding=utf8

import logging
import pickle

import numpy as np
from triflow.misc import materials
from triflow.misc.helpers import init_4f_per
from triflow.plugins import schemes

logger = logging.getLogger()
logger.handlers = []
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# parameters = {}

# physical_parameters = materials.water()
# physical_parameters['Beta'] = np.pi / 2
# physical_parameters['Re'] = 15
# physical_parameters['Bi'] = 2.81E-03
# parameters.update(physical_parameters)

# initial_parameters = {
#     'hini': 1.,
#     'qini': 1 / 3,
#     'sini': 0,
#     'thetaini': physical_parameters['theta_flat'],
#     'phiini': physical_parameters['phi_flat']
# }
# parameters.update(initial_parameters)

# numerical_parameters = {'t': 0, 'dt': 10, 'tmax': 20,
#                         'tol': 1E-1}
# parameters.update(numerical_parameters)

with open('parameter.bin', 'rb') as f:
    parameters = pickle.load(f)
parameters['tol'] = 1E-6
solver, initial_fields = init_4f_per(parameters)

simul = solver.start_simulation(initial_fields, 0, **parameters)
simul.set_scheme(schemes.ROS_mvart_scheme)
next(simul)
next(simul)
