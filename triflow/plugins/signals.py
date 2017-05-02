#!/usr/bin/env python
# coding=utf8
"""This module provides a Signal class usefull for time variable boundary
conditions.

Available signals:
    * ConstantSignal: just an offset signal
    * ForcedSignal: a sinusoidal wave
    * WhiteNoise: a noisy signal
    * BrownNoise: a noisy signal with Fourrier modes set to 0 for a fraction of the available modes.
"""  # noqa

import logging
from functools import partial

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import periodogram, convolve, firwin

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Signal(object):
    """Base class for signal object.

    Args:
        ``signal_period (float): period of the signal.``
        ``n (int, optional): sampling number of the signal.``
        ``**kwargs: extra arguments provided to the custom signal template.``
    """  # noqa

    def __init__(self, signal_period: float, n: int=1000, **kwargs):
        logging.debug(f"{(signal_period, n, kwargs)}")
        self._size = n
        self._signal_period = signal_period
        self._time, self._dt = np.linspace(
            0, self._signal_period, self._size, retstep=True)
        self._template = np.array(self._signal_template(signal_period,
                                                        n=n, **kwargs))
        self._interp_function = self._generate_wave_function(self._time,
                                                             self._template)

    def _signal_template(self):
        """This method has to be overrided by child class with a function which
        return the signal as a numpy.ndarray. The __call__ method use this
        array and an interpolation function to construct the signal.
        Returns:
            np.ndarray: template signal.

        Raises:
            NotImplementedError: is raised if the child class do not override this method.
        """  # noqa
        raise NotImplementedError

    def _generate_wave_function(self, time: np.array,
                                template: np.array):
        return interp1d(time, template)

    def _wave(self, t: float) -> float:
        signal = self._interp_function(t % self._time.max())
        return signal

    def __call__(self, t: float):
        """return the signal value for the time t.

        Args:
            t (float): time where the signal is evaluated.

        Returns:
            float: amplitude of the signal a the time t.
        """

        return self._wave(t)

    def __add__(self, other_signal):
        """
        Args:
            other_signal (triflow.plugins.signal.Signal): second signal to which this one will be added.

        Returns:
            triflow.plugins.signal.AdditiveSignal: added signal.
        """  # noqa
        signal_period = max(self._signal_period, other_signal._signal_period)
        size = max(self._size, other_signal.size)
        return AdditiveSignal(
            self, other_signal, signal_period, size)

    @property
    def fourrier_spectrum(self, *args, **kwargs):
        """
        Returns:
            tuple of numpy.ndarray: fourrier modes and power density obtained with the scipy.signal.periodogram function.

        Args:
            *args, **kwargs: extra arguments provided to the periodogram function.
        """  # noqa
        freq, ampl = periodogram(self._template, fs=1 / self._dt)
        return freq, ampl


class AdditiveSignal(Signal):
    """Additive signal. The two source signals are aligned in order to be added
    """

    def __init__(self, signal_a, signal_b, signal_period, n):
        time = np.linspace(0, signal_period, n)
        self._templates = [signal_a.wave(time), signal_b.wave(time)]
        super().__init__(signal_period, n=n)

    def _signal_template(self, signal_period: float, n: int=1000):
        return np.sum(self._templates, axis=0)


class ConstantSignal(Signal):
    """Offset signal

    Args:
        offset (int, optional): Value of the offset
    """

    def __init__(self, offset):
        super().__init__(1,
                         n=2, offset=offset)

    def _signal_template(self, signal_period: float, n: int=1000,
                         offset=0):
        return 0 * self._time + offset


class GaussianWhiteNoise(Signal):
    """Gaussian White noise signal, seeded with numpy.random.randn

    Args:
        ``fs (float): Frequency sampling (the highest frequency of the white signal spectrum).``
        ``mean (int, optional, default 1000): sampling number of the signal.``
        ``n (int, optional, default 1000): sampling number of the signal.``
        ``n (int, optional, default 1000): sampling number of the signal.``
        ``seed (int or None, optional): pseudo-random number generator seed. Signals with same signal_period, sampling number and seed will be similar.``
    """  # noqa

    def __init__(self,
                 fs,
                 mean=0,
                 std=.2,
                 n=1000,
                 seed=None):
        self._cache = {}
        self._seed = seed
        super().__init__(n / (2 * fs), mean=mean, std=std, seed=seed, n=n)
        self._partial_signal_template = partial(self._signal_template,
                                                n / (2 * fs),
                                                mean=mean, std=std, n=n)

    def _signal_template(self,
                         signal_period,
                         mean,
                         std,
                         seed=None,
                         n: int=1000):
        randgen = np.random.RandomState(seed=seed)
        noisy = randgen.normal(mean, std, size=self._size)
        return noisy

    def _wave(self, t: float or np.array) -> float or np.array:
        ids_iter = np.array(t // self._signal_period, dtype=int)
        signal = np.zeros_like(t)
        for id_iter in ids_iter:
            try:
                interp_function = self._cache[id_iter]
            except KeyError:
                logging.debug("noisy signal outside of the range "
                              f"(iter {id_iter}), reseed.")
                self._cache[id_iter] = self._generate_wave_function(
                    self._time,
                    self._partial_signal_template(seed=(self._seed + id_iter)))
                interp_function = self._cache[id_iter]
            flag = np.where(ids_iter == id_iter)
            signal[flag] = interp_function(t % self._signal_period)
        # logging.debug(signal)
        return signal


class GaussianBrownNoise(GaussianWhiteNoise):
    """Gaussian Brown noise signal, seeded with numpy.random.randn and with frequency cut at specific value.

    Args:
        ``fs (float): Frequency sampling (the highest frequency of the white signal spectrum).``
        ``fcut (float): Filter frequency cut.``
        ``mean (int, optional, default 1000): sampling number of the signal.``
        ``n (int, optional, default 1000): sampling number of the signal.``
        ``n (int, optional, default 1000): sampling number of the signal.``
        ``seed (int or None, optional): pseudo-random number generator seed. Signals with same signal_period, sampling number and seed will be similar.``
    """  # noqa

    def __init__(self,
                 fs,
                 fcut,
                 mean=0,
                 std=.2,
                 n=1000,
                 filter_std=25,
                 filter_window_size=500,
                 seed=None):
        self._cache = {}
        self._seed = seed
        super(GaussianWhiteNoise, self).__init__(
            n / (2 * fs),
            fs=fs, fcut=fcut,
            mean=mean, std=std,
            seed=seed, n=n,
            filter_std=filter_std,
            filter_window_size=filter_window_size,)
        self._partial_signal_template = partial(
            self._signal_template,
            n / (2 * fs), fs=fs, fcut=fcut,
            mean=mean, std=std, n=n,
            filter_std=filter_std,
            filter_window_size=filter_window_size,)

    def _signal_template(self,
                         signal_period,
                         fs,
                         fcut,
                         mean,
                         std,
                         filter_std=25,
                         filter_window_size=500,
                         seed=None,
                         n: int=1000):
        randgen = np.random.RandomState(seed=seed)
        noisy = randgen.normal(mean, std, size=self._size)
        high_band_filter = firwin(filter_window_size, fcut,
                                  nyq=fs, window=('gaussian', filter_std))
        filtered_noisy = convolve(noisy, high_band_filter, mode='same')
        return filtered_noisy


class SinusoidalSignal(Signal):
    """Summary

    Args:
        ``signal_period (float): Period on which the signal will be repeated.``
        ``n (int, optional, default 1000): sampling number of the signal.``
        ``noise_ampl (float, optional, default 0.1): amplitude of the signal.``
        ``signal_ampl (float, optional): Description``
    """  # noqa

    def __init__(self, frequency, amplitude, phase=0, n: int=1000):
        # we take a period > 2 sinusoidal period to avoid aliasing.
        super().__init__(3 / frequency, n=n,
                         frequency=frequency, amplitude=amplitude, phase=phase)
        freq, ampl = self.fourrier_spectrum
        if not np.isclose(frequency, freq[np.argmax(ampl)],
                          rtol=1E-3, atol=1E-5):
            logging.warning('There is difference between '
                            'chosen frequency (%.5e) and '
                            'main frequency of the fourrier '
                            'spectrum (%.5e)\n'
                            'They may have some aliasing, '
                            'you should increase the number of samples '
                            '(n parameter, see signal doc)' %
                            (frequency, freq[np.argmax(ampl)]))

    def _signal_template(self, signal_period, frequency, amplitude, phase,
                         n):
        return amplitude * np.sin(self._time * 2 * np.pi *
                                  frequency + phase)
