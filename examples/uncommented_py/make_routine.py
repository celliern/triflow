#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
from triflow.misc.helpers import init_2f_open
from triflow.core.make_routines import make_routines_fortran, cache_routines_fortran
from triflow.models.boundaries import periodic_boundary
from triflow.models.model_4fields import model
from triflow.misc.materials import water

logger = logging.getLogger()
logger.handlers = []
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# parameters = {}

# physical_parameters = water()
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

# numerical_parameters = {'t': 0, 'dt': 100, 'tmax': 200,
#                         'tol': 1E-1}
# parameters.update(numerical_parameters)

# domain_parameters = {'L': 1000, 'Nx': 1400}
# parameters.update(domain_parameters)

# solver, initial_fields = init_4f_open(parameters)
# simul = solver.start_simulation(initial_fields, 0, **parameters)
# simul.solver.compute_J = profile(simul.solver.compute_J)
# simul.solver.compute_J(simul.U, **simul.pars)

cache_routines_fortran(model, periodic_boundary, 'test')
