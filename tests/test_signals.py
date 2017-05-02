#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import numpy as np
# import pytest

from triflow.plugins import signals


def test_sin_signal():
    signals.SinusoidalSignal(1, 1)


def test_white_signal():
    signals.GaussianWhiteNoiseWhiteNoise(200)


def test_brown_signals():
    signals.GaussianBrownNoise(200, 50)
