#!/usr/bin/env python
# coding=utf8

import logging

from triflow.make_routines import (cache_routines_fortran,
                                   load_routines_fortran,
                                   make_routines_fortran)
from triflow.simulation import Simulation
from triflow.solver import Solver

try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
