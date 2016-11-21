#!/usr/bin/env python
# coding=utf8

import logging

from triflow.core import make_routines, simulation, solver
from triflow.models import boundaries

from triflow.core.make_routines import (cache_routines_fortran,
                                        load_routines_fortran,
                                        make_routines_fortran)
from triflow.core.simulation import Simulation
from triflow.core.solver import Solver

try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):

        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
