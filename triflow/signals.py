#!/usr/bin/env python
# coding=utf8

from scipy.fftpack import rfft, irfft
from scipy.interpolate import interp1d
import numpy as np


class Signal(object):
    def __init__(self, simul, **kwargs):
        self.simul = simul
        self.size = simul.pars.get('signal_size', 10000)
        self.tmax = simul.pars['tmax']
        self.time_period = np.linspace(0, self.tmax, self.size)
        kwargs.update(simul.pars)
        self.template = np.array(self.signal_template(**kwargs))
        self.generate_wave_function()

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
        templates = self.template, other_signal.template
        time_periodes = self.time_period, other_signal.time_period
        assert np.allclose(*time_periodes), "Incompatible time signals"
        return type('Signal', (Signal,),
                    {'signal_template':
                     lambda self, **kwarg:
                     np.sum(templates, axis=0)})(self.simul)


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


def init_signal(simul, field, Signals=(NoNoise,)):
    signal = sum([Signal(simul.pars) for Signal in Signals])

    def frequencies(simul, t):
        simul.pars['h%' % field] = signal(t)
    return frequencies


if __name__ == '__main__':
    import pylab as pl

    signal1 = WhiteNoise(10000, noise_ampl=.01)
    signal2 = BrownNoise(10000, noise_ampl=.01, fcut=.3)
    signal3 = ForcedSignal(10000, signal_freq=10, signal_ampl=.1,
                           offset=1.)
    sum_signal_white = signal1 + signal3
    sum_signal_brown = signal2 + signal3

    pl.subplot(321)
    pl.plot(signal1(signal1.time_period))
    pl.subplot(323)
    pl.plot(signal3(signal1.time_period))
    pl.subplot(325)
    pl.plot(sum_signal_white(signal1.time_period))

    pl.subplot(322)
    pl.plot(signal2(signal1.time_period))
    pl.subplot(324)
    pl.plot(signal3(signal1.time_period))
    pl.subplot(326)
    pl.plot(sum_signal_brown(signal1.time_period))
    pl.show()
