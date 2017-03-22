#!/usr/bin/env python
# coding=utf8

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hamming
from triflow.plugins import signals


def make_initial_wave(x, ampl, offset):
    return hamming(x.size) * ampl + offset - ampl / 2


def make_jump(x):
    U = ((-(np.tanh(
        (x - .1 * x[-1]) / 20) + 1) / 2 + 1) * 1 + 1) / 2
    return U


class NoisyHook:
    def __init__(self, freq, turb=False):
        sin_sig = signals.ForcedSignal(1 / freq,
                                       signal_freq=freq,
                                       signal_ampl=.1)
        cst_sig = signals.ConstantSignal(1 / freq, offset=1)
        noisy_sig = signals.BrownNoise(1 / freq, noise_ampl=.1,
                                       seed=74897497)
        self.sig = sin_sig + cst_sig + noisy_sig
        self.turb = turb

    def __call__(self, fields, t, pars):
        fields.h[0] = self.sig(t)
        fields.q[0] = self.sig(t) if self.turb else self.sig(t) ** 3 / 3

        return fields, pars


class TurbCallable:
    def __init__(self, inter):
        self.inter = inter

    def __call__(self, fields, t, pars):
        q = fields.q
        inter = self.inter
        for key, _ in zip(*inter.trad):
            fields[key][:] = inter.get(key, q)
        return fields, pars


class Interpolator:
    def __init__(self, data_path, Re):
        data = dict(**np.load(data_path))
        to_interpolate = {key: value for key, value in data.items()
                          if key not in ('grid_q', 'grid_Re')}
        trad = dict(zip(["cK", "cM", "cJ",
                         "cL", "cF", "cG", "cS",
                         "Cthth", "Cthphi", "Cphith", "Cphiphi",
                         "bita", "bita1", "betath", "betaphi",
                         "ffath", "thetaz", "phiz"],
                        ['K(q)', 'M(q)', 'J(q)',
                         'L(q)', 'F(q)', 'G(q)', 'S(q)',
                         'Cthth', 'Cthphi', 'Cphith', 'Cphiphi',
                         'Bita(q)', 'Bita1', 'betath', 'betaphi',
                         'ffath', 'thetaz', 'phiz']))

        self.data = data
        self.interpRe = {key: interp1d(data['grid_Re'],
                                       value,
                                       axis=0,
                                       bounds_error=False)(Re)
                         for key, value in to_interpolate.items()}
        self.interp = {key: interp1d(self.data['grid_q'],
                                     self.interpRe[trad[key]],
                                     bounds_error=False,
                                     fill_value=(
            self.interpRe[trad[key]][0],
            self.interpRe[trad[key]][-1]
        ))
            for key in trad}
        self.interp['Fr'] = interp1d(self.data['grid_q'],
                                     self.interpRe['Fr'],
                                     bounds_error=False,
                                     fill_value=(self.interpRe['Fr'][0],
                                                 self.interpRe['Fr'][-1]))
        self.trad = trad

    def get(self, key, q):
        return self.interp[key](q)
