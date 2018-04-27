#!/usr/bin/env python
# coding=utf8

import numpy as np
import pytest

from triflow import Model, Simulation, display_fields, display_probe


@pytest.fixture
def heat_model():
    model = Model(differential_equations="k * dxxT",
                  dependent_variables="T",
                  parameters="k", compiler="numpy")
    return model


def test_display_fields(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    simul = Simulation(heat_model, initial_fields, parameters,
                       dt=1, tmax=2, tol=1E-1)
    display_fields(simul)
    for i, (t, fields) in enumerate(simul):
        continue


def test_display_probes(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    simul = Simulation(heat_model, initial_fields, parameters,
                       dt=1, tmax=2, tol=1E-1)
    display_probe(simul, function=lambda simul: simul.timer.total)
    for i, (t, fields) in enumerate(simul):
        continue
