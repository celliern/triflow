#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
from triflow.core.make_routines import cache_routines_fortran, load_routines_fortran
from triflow.models.boundaries import periodic_boundary
from triflow.models.model_full_fourrier import model as fmodel
from triflow.models.model_2fields import model as smodel
from triflow.core.solver import Solver

logger = logging.getLogger()
logger.handlers = []
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

cache_routines_fortran(lambda: smodel(),
                       periodic_boundary, 'test')
solver = Solver('test')
