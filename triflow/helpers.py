#!/usr/bin/env python
# coding=utf8

import numpy as np
from scipy.signal import hamming
from triflow.solver import Solver


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
    Ti = parameters['theta_flat'] * y[np.newaxis, :] + x[:, np.newaxis] * 0
    solver = Solver('%iff_per' % Ny)
    return solver, [hi, qi, *Ti.T]


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
    Ti = parameters['theta_flat'] * y[np.newaxis, :] + x[:, np.newaxis] * 0
    solver = Solver('%iff_per' % Ny)
    return solver, [hi, qi, *Ti.T]
