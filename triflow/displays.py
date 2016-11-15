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
from triflow.misc import coroutine


@coroutine
def none_display(simul):
    while True:
        simul = yield


@coroutine
def simple_display(simul):
    while True:
        simul = yield simul.t, simul.solver.get_fields(simul.U)


@coroutine
def full_display(simul):
    time_list = [simul.t]
    sol_list = np.array([simul.solver.get_fields(simul.U)])
    simul = yield time_list, sol_list
    while True:
        time_list.append(simul.t)
        sol_list = np.vstack([sol_list, simul.solver.get_fields(simul.U)])
        simul = yield time_list, sol_list
