#!/usr/bin/env python
# coding=utf8

from scipy.signal import hamming

import numpy as np
from triflow.core.make_routines import cache_routines_fortran
from triflow.core.solver import Solver
from triflow.models.boundaries import openflow_boundary, periodic_boundary
from triflow.models.model_4fields import model
from triflow.models.model_full_fourrier import model as ffmodel


def init_2f_per(parameters):
    x, dx = np.linspace(
        0, parameters['L'], parameters['Nx'], endpoint=False, retstep=True)
    hi = hamming(parameters['Nx']) * .1 + .95
    qi = hi**3 / 3
    s = x * 0
    solver = Solver('2fields_per')
    return solver, [hi, qi, s]


def init_4f_per(parameters):
    x, dx = np.linspace(
        0, parameters['L'], parameters['Nx'], endpoint=False, retstep=True)
    hi = hamming(parameters['Nx']) * .1 + .95
    qi = hi**3 / 3
    thetai = x * 0 + parameters['theta_flat']
    phii = x * 0 + parameters['phi_flat']
    s = x * 0
    solver = Solver('4fields_per')
    return solver, [hi, qi, thetai, phii, s]


def init_ff_per(Ny, parameters):
    x, dx = np.linspace(
        0, parameters['L'], parameters['Nx'], endpoint=False, retstep=True)
    # Les champs de températures sont répartis sur des points de GLC.
    y = ((-np.cos(np.pi * np.arange(Ny) / (Ny - 1))) + 1) / 2
    hi = hamming(parameters['Nx']) * .1 + .95
    qi = hi**3 / 3
    Ti = ((parameters['theta_flat'] - 1) *
          y[np.newaxis, :] + 1 + x[:, np.newaxis] * 0)
    solver = Solver('%iff_per' % Ny)
    return solver, [hi, qi, *Ti.T]


def init_2f_open(parameters):
    x, dx = np.linspace(
        0, parameters['L'], parameters['Nx'], endpoint=False, retstep=True)
    hi = ((-(np.tanh(
        (x - .1 * parameters['L']) / 20) + 1) / 2 + 1) * 1 + 1) / 2
    qi = hi**3 / 3
    s = x * 0
    solver = Solver('2fields_open')
    return solver, [hi, qi, s]


def init_4f_open(parameters):
    x, dx = np.linspace(
        0, parameters['L'], parameters['Nx'], endpoint=False, retstep=True)
    # La tangente hyperbolique permet de faire passer la première vague
    # en limitant le raideur du problème.
    hi = ((-(np.tanh(
        (x - .1 * parameters['L']) / 20) + 1) / 2 + 1) * 1 + 1) / 2
    qi = hi**3 / 3
    thetai = x * 0 + parameters['theta_flat']
    phii = x * 0 + parameters['phi_flat']
    s = x * 0
    solver = Solver('4fields_open')
    return solver, [hi, qi, thetai, phii, s]


def init_ff_open(Ny, parameters):
    x, dx = np.linspace(
        0, parameters['L'], parameters['Nx'], endpoint=False, retstep=True)
    # Les champs de températures sont répartis sur des points de GLC.
    y = ((-np.cos(np.pi * np.arange(Ny) / (Ny - 1))) + 1) / 2
    # La tangente hyperbolique permet de faire passer la première vague
    # en limitant le raideur du problème.
    hi = ((-(np.tanh(
        (x - .1 * parameters['L']) / 20) + 1) / 2 + 1) * 1 + 1) / 2
    qi = hi**3 / 3
    Ti = ((parameters['theta_flat'] - 1) *
          y[np.newaxis, :] + 1 + x[:, np.newaxis] * 0)
    print(Ti)
    solver = Solver('%iff_per' % Ny)
    return solver, [hi, qi, *Ti.T]
