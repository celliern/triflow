#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from triflow.plugins import signals


def test_notimplemented_signal():
    with pytest.raises(NotImplementedError):
        signals.Signal(5)


def test_sin_signal():
    signals.SinusoidalSignal(1, 1)


def test_white_signal():
    signals.GaussianWhiteNoise(200, )


def test_brown_signals():
    signals.GaussianBrownNoise(200, 50)


def test_add():
    t = np.linspace(0, 1, 1000)
    noisy = signals.GaussianWhiteNoise(frequency_sampling=200)
    offset = signals.ConstantSignal(offset=1)
    noise_with_offset = noisy + offset
    noise_without_offset = noise_with_offset - offset
    np.isclose(noisy(t), noise_without_offset(t)).all()
    signals.GaussianWhiteNoise(frequency_sampling=200)


def test_white():
    t = np.linspace(0, 10, 1000)
    signal1 = signals.GaussianWhiteNoise(frequency_sampling=200, seed=50)
    signal2 = signals.GaussianWhiteNoise(frequency_sampling=200, seed=50)
    assert np.isclose(signal1(t) - signal2(t), 0).all()
    signal3 = signals.GaussianWhiteNoise(frequency_sampling=200)
    assert not np.isclose(signal1(t) - signal3(t), 0).all()


def test_brown():
    brown_signal = signals.GaussianBrownNoise(frequency_sampling=200,
                                              frequency_cut=50)
    spectrum_frequencies, spectrum_density = brown_signal.fourrier_spectrum
    assert np.isclose(spectrum_density[spectrum_frequencies > 50],
                      0, atol=1E-3).all()


def test_sin():
    from triflow.plugins import signals
    t = np.linspace(0, 100, 10000)
    signal = signals.SinusoidalSignal(frequency=5, amplitude=.5)
    spectrum_frequencies, spectrum_density = signal.fourrier_spectrum
    assert np.isclose(signal(t).max(), 0.5)
    assert np.isclose(spectrum_frequencies[spectrum_density.argmax()], 4.995)
    with pytest.warns(UserWarning):
        signals.SinusoidalSignal(5, 1, n=30)
