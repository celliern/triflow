#!/usr/bin/env python
# coding=utf8

import logging
import multiprocessing as mp
from logging import info

from triflow.core.make_routines import cache_routines_fortran
from triflow.models.boundaries import openflow_boundary, periodic_boundary
from triflow.models.model_2fields import model as model2
from triflow.models.model_4fields import model as model4
from triflow.models.model_full_fourrier import model as modelfull

logger = logging.getLogger()
logger.handlers = []
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def cache_full():
    processes = []

    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(lambda: modelfull(10),
                                      periodic_boundary,
                                      '%iff_per' % 10)))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(lambda: modelfull(10),
                                      openflow_boundary,
                                      '%iff_open' % 10)))

    for process in processes:
        process.start()

    finish = False
    while not finish:
        finish = True
        for process in processes:
            finish *= not process.is_alive()
    info("processes are finish")

    for process in processes:
        process.join()


def cache_simple():
    processes = []

    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model2, openflow_boundary,
                                      '2fields_open')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model4, openflow_boundary,
                                      '4fields_open')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model2, periodic_boundary,
                                      '2fields_per')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model4, periodic_boundary,
                                      '4fields_per')))

    for process in processes:
        process.start()

    finish = False
    while not finish:
        finish = True
        for process in processes:
            finish *= not process.is_alive()
    info("processes are finish")

    for process in processes:
        process.join()
