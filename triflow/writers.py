#!/usr/bin/env python
# coding=utf8

import numpy as np
from triflow.misc import coroutine


@coroutine
def datreant_last_step_writer(simul, path=None):
    while True:
        simul = yield


@coroutine
def datreant_all_step_writer(simul, path=None):
    while True:
        simul = yield


@coroutine
def bokeh_writer(simul, fields=None):
    while True:
        simul = yield
