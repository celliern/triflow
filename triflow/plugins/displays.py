#!/usr/bin/env python
# coding=utf8

"""
Displays are written as coroutine: they take the simulation state and
return what we want.

It actually do not need the full simulation but any object with the
needed attribute (usually t and U).

The coroutine give the ability to initialize the function
(as seen in full display).
"""

import numpy as np


def none_display(simul):
    while True:
        yield


def simple_display(simul):
    while True:
        yield simul.t, simul.solver.get_fields(simul.U)


def simple_display_with_id(simul):
    while True:
        yield simul.t, simul.solver.get_fields(simul.U), simul.pars


def full_display(simul):
    time_list = [simul.t]
    sol_list = np.array([simul.solver.get_fields(simul.U)])
    yield time_list, sol_list
    while True:
        time_list.append(simul.t)
        sol_list = np.vstack([sol_list, simul.solver.get_fields(simul.U)])
        yield time_list, sol_list
