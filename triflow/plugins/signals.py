#!/usr/bin/env python
# coding=utf8

import copy
from scipy.fftpack import rfft, irfft
from scipy.interpolate import interp1d
import numpy as np


class Signal(object):
    def __init__(self, **kwargs):
        self.size = kwargs.get('signal_size', 10000)
        self.tmax = kwargs['tmax']
        self.time_period = np.linspace(0, self.tmax, self.size)
        self.template = np.array(self.signal_template(**kwargs))
        self.generate_wave_function()
        self.pars = kwargs

    def signal_template(self, **kwarg):
        raise NotImplementedError

    def generate_wave_function(self):
        self.interp_function = interp1d(self.time_period, self.template)

    def wave(self, t):
        signal = self.interp_function(t)
        return signal

    def __call__(self, t):
        return self.wave(t % self.tmax)

    def __add__(self, other_signal):
        time_periodes = self.time_period, other_signal.time_period
        pars = self.pars.copy()
        pars.update(other_signal.pars)
        assert np.allclose(*time_periodes), "Incompatible time signals"
        return AdditiveSignal(self, other_signal, **pars)


class AdditiveSignal(Signal):
    def __init__(self, signal_a, signal_b, **kwarg):
        self.templates = [signal_a.template, signal_b.template]
        super().__init__(**kwarg)

    def signal_template(self, **kwarg):
        return np.sum(self.templates, axis=0)


class NoNoise(Signal):
    def signal_template(self, **kwarg):
        return 0 * self.time_period


class BrownNoise(Signal):
    def signal_template(self, noise_ampl=.1, fcut=.5, offset=0, **kwarg):
        input_modes = rfft(np.random.rand(self.size) * 2 - 1)
        input_modes[int(fcut * self.size):] = 0
        return noise_ampl * irfft(input_modes) + offset


class WhiteNoise(Signal):
    def signal_template(self, noise_ampl=.1, offset=0, **kwarg):
        input_modes = rfft(np.random.rand(self.time_period.size) * 2 - 1)
        return noise_ampl * irfft(input_modes) + offset


class ForcedSignal(Signal):
    def signal_template(self, signal_freq=1, signal_ampl=.1,
                        signal_phase=0, offset=0., **kwarg):
        return signal_ampl * np.sin(self.time_period * 2 * np.pi *
                                    signal_freq + signal_phase) + offset


# def init_signal(simul, field, Signals=(NoNoise,)):
#     signal = sum([Signal(simul.pars) for Signal in Signals])

#     def frequencies(simul, t):
#         simul.pars['h%' % field] = signal(t)
#     return frequencies
