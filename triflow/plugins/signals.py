#!/usr/bin/env python
# coding=utf8

import logging
from scipy.fftpack import rfft, irfft
from scipy.interpolate import interp1d
from scipy.signal import periodogram
from typing import Callable
import numpy as np
from toolz.functoolz import memoize

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Signal(object):
    def __init__(self, signal_period: float, n: int=1000, **kwargs):
        self.size = n
        self.signal_period = signal_period
        self.time, self.dt = np.linspace(
            0, self.signal_period, self.size, retstep=True)
        self.template = np.array(self._signal_template(**kwargs))
        self.interp_function = self.generate_wave_function(self.time,
                                                           self.template)

    def _signal_template(self, **kwarg):
        raise NotImplementedError

    def generate_wave_function(self, time: np.array,
                               template: np.array) -> Callable:
        return interp1d(time, template)

    def wave(self, t: float) -> float:
        signal = self.interp_function(t % self.signal_period)
        return signal

    def __call__(self, t: float) -> float:
        return self.wave(t)

    def __add__(self, other_signal):
        signal_period = max(self.signal_period, other_signal.signal_period)
        size = max(self.size, other_signal.size)
        return AdditiveSignal(
            self, other_signal, signal_period=signal_period, n=size)

    @property
    def fourrier_spectrum(self):
        freq, ampl = periodogram(self.template, fs=1 / self.dt)
        return freq, ampl


class AdditiveSignal(Signal):
    def __init__(self, signal_a, signal_b, signal_period, n):
        time = np.linspace(0, signal_period, n)
        self.templates = [signal_a.wave(time), signal_b.wave(time)]
        super().__init__(signal_period, n=n)

    def _signal_template(self):
        return np.sum(self.templates, axis=0)


class ConstantSignal(Signal):
    def __init__(self, signal_period: float, n: int=1000, offset=0, **kwargs):
        super().__init__(offset=offset)

    def _signal_template(self, offset=0, **kwargs):
        return 0 * self.time_period + offset


class BrownNoise(Signal):
    def __init__(self,
                 signal_period: float, n: int=1000,
                 noise_ampl: float=.01,
                 fcut: float=.5,
                 offset: float=0,
                 seed: int or None=None,
                 **kwargs):
        super().__init__(
            noise_ampl=noise_ampl,
            fcut=fcut,
            offset=offset,
            seed=seed)

    @memoize
    def _signal_template(self,
                         signal_period: float, n: int=1000,
                         noise_ampl: float=.01,
                         fcut: float=.5,
                         offset: float=0,
                         seed: int or None=None,
                         **kwargs):
        randgen = np.random.RandomState(seed=seed)
        input_modes = rfft(randgen.rand(self.size) * 2 - 1)
        if fcut is not None:
            input_modes[int(fcut * self.size):] = 0
        return noise_ampl * irfft(input_modes) + offset

    def wave(self, t: float or np.array) -> float or np.array:
        signal = self.interp_function(t % self.signal_period)
        return signal


class WhiteNoise(BrownNoise):
    def __init__(self,
                 signal_period: float, n: int=1000,
                 noise_ampl: float=.01,
                 offset: float=0,
                 seed: int or None=None,
                 **kwargs):
        super().__init__(
            noise_ampl=noise_ampl,
            fcut=None,
            offset=offset,
            seed=seed)


class ForcedSignal(Signal):
    def __init__(self, signal_period: float, n: int=1000,
                 signal_freq: float=1, signal_ampl: float=.1, **kwarg):
        super().__init__(self, signal_period, n,
                         signal_freq=signal_freq, signal_ampl=signal_ampl)
        freq, ampl = self.fourrier_spectrum
        if not np.isclose(kwarg['freq'], freq[np.argmax(ampl)], rtol=.01):
            logging.warning('There is difference between '
                            'chosen frequency (%.3e) and '
                            'main frequency of the fourrier '
                            'spectrum (%.3e)\n'
                            'They may have some aliasing, '
                            'you should increase the number of samples '
                            '(n parameter, see signal doc)' %
                            (kwarg['freq'], freq[np.argmax(ampl)]))

    def _signal_template(self, freq=1, ampl=.1, phase=0, offset=0., **kwarg):
        return ampl * np.sin(self.time * 2 * np.pi * freq + phase) + offset
