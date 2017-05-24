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
import warnings
from functools import partial

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve, firwin, periodogram

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.captureWarnings(True)
logging = logging.getLogger(__name__)


class Signal(object):
    """Base class for signal object.

      Parameters
      ----------
      signal_period : float
          period of the signal.
      n : int, optional
          sampling number of the signal.
      **kwargs
          extra arguments provided to the custom signal template.

      Raises
      ------
      NotImplementedError
      this class is not supposed to be used by the user. If initialized directly, it will raise a NotImplementedError.
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

    def _signal_template(self, *args, **kwargs):
        """This method has to be overrided by child class with a function which
          return the signal as a numpy.ndarray. The __call__ method use this
          array and an interpolation function to construct the signal.

          Returns
          -------
          np.ndarray
              template signal.

          Raises
          ------
          NotImplementedError
              is raised if the child class do not override this method.

          Parameters
          ----------
          *args
              Description
          **kwargs
              Description
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

        Parameters
        ----------
        t : float
            time where the signal is evaluated.

        Returns
        -------
        float
            amplitude of the signal a the time t.
        """

        return self._wave(t)

    def __add__(self, other_signal):
        """
          Parameters
          ----------
          other_signal : triflow.plugins.signal.Signal
              second signal to which this one will be added.

          Returns
          -------
          triflow.plugins.signal.AdditiveSignal
              added signal.

          """  # noqa

        return AdditiveSignal(self, other_signal, op=np.add)

    def __sub__(self, other_signal):
        """
          Parameters
          ----------
          other_signal : triflow.plugins.signal.Signal
              second signal to which this one will be added.

          Returns
          -------
          triflow.plugins.signal.AdditiveSignal
              added signal.

          """  # noqa

        return AdditiveSignal(self, other_signal, op=np.subtract)

    @property
    def fourrier_spectrum(self, *args, **kwargs):
        """
          Parameters
          ----------
          *args, **kwargs
              extra arguments provided to the periodogram function.

          Returns
          -------
          tuple of numpy.ndarray: fourrier modes and power density obtained with the scipy.signal.periodogram function.
          """  # noqa
        freq, ampl = periodogram(self._template, fs=1 / self._dt)
        return freq, ampl


class AdditiveSignal(Signal):
    """Additive signal. Proxy the different signals and sum their interpolation functions.
      """  # noqa

    def __init__(self, signal_a, signal_b, op):
        self._signal_list = []
        for signal in signal_a, signal_b:
            if isinstance(signal, AdditiveSignal):
                self._signal_list += signal._signal_list
            else:
                self._signal_list.append((signal, op))
        signal_period = max([signal[0]._signal_period
                             for signal in
                             self._signal_list])
        n = max([signal[0]._size
                 for signal in
                 self._signal_list])
        super().__init__(signal_period, n=n)

    def _signal_template(self, signal_period: float, n: int=1000):
        return None

    def _generate_wave_function(self, time: np.array,
                                template: np.array):
        return None

    def _wave(self, t: float) -> float:
        total_signal = 0
        for signal, op in self._signal_list:
            total_signal = op(total_signal,
                              signal._interp_function(t % signal._time.max()))
        return total_signal


class ConstantSignal(Signal):
    """Offset signal

    Parameters
    ----------
    offset : float
        Value of the offset

    Examples
    --------
    Set an offset to a white noise:

    >>> import numpy as np
    >>> from triflow.plugins import signals
    >>> t = np.linspace(0, 1, 1000)
    >>> noisy = signals.GaussianWhiteNoise(frequency_sampling=200)
    >>> offset = signals.ConstantSignal(offset=1)
    >>> noise_with_offset = noisy + offset
    >>> noise_without_offset = noise_with_offset - offset
    >>> np.isclose(noisy(t), noise_without_offset(t)).all()
    True
    """

    def __init__(self, offset: float):
        super().__init__(1,
                         n=2, offset=offset)

    def _signal_template(self, signal_period: float, n: int=1000,
                         offset=0):
        return 0 * self._time + offset


class GaussianWhiteNoise(Signal):
    """Gaussian White noise signal, seeded with numpy.random.randn

      Parameters
      ----------
      frequency_sampling : float
          Frequency sampling (the highest frequency of the white signal spectrum).
      mean : int, optional, default 1000
          sampling number of the signal.
      n : int, optional, default 1000
          sampling number of the signal.
      seed : int or None, optional
          pseudo-random number generator seed. Signals with same signal_period, sampling number and seed will be similar.

      Examples
      --------
      Simple white signal with a 200 Hz frequency sampling:

      >>> import numpy as np
      >>> from triflow.plugins import signals
      >>> signal = signals.GaussianWhiteNoise(frequency_sampling=200)

      Forcing the pseudo-random-number generator seed for reproductible signal:

      >>> import numpy as np
      >>> from triflow.plugins import signals
      >>> t = np.linspace(0, 10, 1000)
      >>> signal1 = signals.GaussianWhiteNoise(frequency_sampling=200, seed=50)
      >>> signal2 = signals.GaussianWhiteNoise(frequency_sampling=200, seed=50)
      >>> np.isclose(signal1(t) - signal2(t), 0).all()
      True
      >>> signal3 = signals.GaussianWhiteNoise(frequency_sampling=200)
      >>> np.isclose(signal1(t) - signal3(t), 0).all()
      False
      """  # noqa

    def __init__(self,
                 frequency_sampling,
                 mean=0,
                 std=.2,
                 n=1000,
                 seed=None):
        self._cache = {}
        self._seed = seed
        super().__init__(n / (2 * frequency_sampling), mean=mean, std=std,
                         seed=seed, n=n)
        self._partial_signal_template = partial(self._signal_template,
                                                n / (2 * frequency_sampling),
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
        if not self._seed:
            return self._interp_function(t % self._signal_period)
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

      Parameters
      ----------
      frequency_sampling : float
          Frequency sampling (the highest frequency of the white signal spectrum).
      frequency_cut : float
          Filter frequency cut.
      mean : int, optional, default 1000
          sampling number of the signal.
      n : int, optional, default 1000
          sampling number of the signal.
      seed : int or None, optional
          pseudo-random number generator seed. Signals with same signal_period, sampling number and seed will be similar.

      Examples
      --------
      White signal with frequencies cut after 50 Hz:

      >>> from triflow.plugins import signals
      >>> brown_signal = signals.GaussianBrownNoise(frequency_sampling=200,
      ...                                           frequency_cut=50)
      >>> spectrum_frequencies, spectrum_density = brown_signal.fourrier_spectrum
      >>> np.isclose(spectrum_density[spectrum_frequencies > 50], 0, atol=1E-3).all()
      True
      """  # noqa

    def __init__(self,
                 frequency_sampling: float,
                 frequency_cut: float,
                 mean=0,
                 std=.2,
                 n=1000,
                 filter_std=25,
                 filter_window_size=500,
                 seed=None):
        self._cache = {}
        self._seed = seed
        super(GaussianWhiteNoise, self).__init__(
            n / (2 * frequency_sampling),
            frequency_sampling=frequency_sampling, frequency_cut=frequency_cut,
            mean=mean, std=std,
            seed=seed, n=n,
            filter_std=filter_std,
            filter_window_size=filter_window_size,)
        self._partial_signal_template = partial(
            self._signal_template,
            n / (2 * frequency_sampling),
            frequency_sampling=frequency_sampling,
            frequency_cut=frequency_cut,
            mean=mean, std=std, n=n,
            filter_std=filter_std,
            filter_window_size=filter_window_size,)

    def _signal_template(self,
                         signal_period,
                         frequency_sampling,
                         frequency_cut,
                         mean,
                         std,
                         filter_std=25,
                         filter_window_size=500,
                         seed=None,
                         n: int=1000):
        randgen = np.random.RandomState(seed=seed)
        noisy = randgen.normal(mean, std, size=self._size)
        high_band_filter = firwin(filter_window_size, frequency_cut,
                                  nyq=frequency_sampling,
                                  window=('gaussian', filter_std))
        filtered_noisy = convolve(noisy, high_band_filter, mode='same')
        return filtered_noisy


class SinusoidalSignal(Signal):
    """Simple sinusoidal signal

      Parameters
      ----------
      frequency : float
          frequency of the signal
      amplitude : float
          amplitude of the signal
      phase : float, optional
          phase of the signam
      n : int, optional
          number of sample for 2 period of the signal

      Examples
      --------
      We generate a 5 Hz signal with an amplitude of 0.5, and we check the signal.

      >>> from triflow.plugins import signals
      >>> t = np.linspace(0, 100, 10000)
      >>> signal = signals.SinusoidalSignal(frequency=5, amplitude=.5)
      >>> spectrum_frequencies, spectrum_density = signal.fourrier_spectrum
      >>> print(f"Amplitude: {signal(t).max():g}")
      Amplitude: 0.499999
      >>> print(f"main mode: {spectrum_frequencies[spectrum_density.argmax()]:g} Hz")
      main mode: 4.995 Hz

      The class give a warning if the number of sample is too low and lead to aliasing. You can try it with

      >>> import logging
      >>> logger = logging.getLogger('triflow.plugins.signals')
      >>> logger.handlers = []
      >>> logger.addHandler(logging.StreamHandler())
      >>> logger.setLevel('INFO')
      >>> aliased_signal = signals.SinusoidalSignal(5, 1, n=30)
      """  # noqa

    def __init__(self,
                 frequency: float,
                 amplitude: float,
                 phase: float=0, n: int=1000):

        # we take a period = 2 sinusoidal period to avoid aliasing.
        super().__init__(2 / frequency, n=n,
                         frequency=frequency, amplitude=amplitude, phase=phase)
        freq, ampl = self.fourrier_spectrum
        if not np.isclose(frequency, freq[np.argmax(ampl)],
                          rtol=1E-3, atol=1E-5):
            warnings.warn('There is difference between '
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
