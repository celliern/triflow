#!/usr/bin/env python
# coding=utf8

from scipy.signal import hamming
import numpy as np


def make_initial_wave(x, ampl, offset):
    return hamming(x.size) * ampl + offset - ampl / 2


def make_jump(x):
    U = ((-(np.tanh(
        (x - .1 * x[-1]) / 20) + 1) / 2 + 1) * 1 + 1) / 2
    return U
