#!/usr/bin/env python
# coding=utf8

import os

import matplotlib as mpl
import numpy as np
import pytest

from triflow import Model, Simulation, display_fields, display_probe  # noqa

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


@pytest.fixture
def heat_model():
    model = Model(
        evolution_equations="k * dxxT",
        dependent_variables="T",
        parameters="k",
        compiler="theano")
    return model


@pytest.fixture
def simul(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T, k=1)
    simul = Simulation(heat_model, initial_fields,
                       dt=.5, tmax=2, tol=1E-1)
    return simul


def test_display_fields(simul):
    display_fields(simul)
    simul.run()


def test_display_probes(simul):
    display_probe(simul, function=lambda simul: simul.timer.total)
    simul.run()


def test_display_mul(simul):
    display_fields(simul) * display_fields(simul)
    display_fields(simul) * display_fields(simul).hv_curve


def test_display_add(simul):
    display_fields(simul) + display_fields(simul)
    display_fields(simul) + display_fields(simul).hv_curve


def test_display_api(simul):
    display = display_fields(simul)
    display.hv_curve
