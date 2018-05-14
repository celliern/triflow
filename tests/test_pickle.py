#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pickle

from triflow import Model, Simulation


@pytest.fixture
def heat_model():
    model = Model(differential_equations="k * dxxT",
                  dependent_variables="T",
                  parameters="k", compiler="numpy")
    return model


def test_pickling(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    simul = Simulation(heat_model, initial_fields,
                       parameters, dt=1, tmax=100,
                       tol=1E-1)
    container = simul.attach_container("/tmp/test")

    pickle.loads(pickle.dumps(heat_model))
    pickle.loads(pickle.dumps(container))
    pickle.loads(pickle.dumps(simul))

    simul.run()

    pickle.loads(pickle.dumps(container))
    pickle.loads(pickle.dumps(simul))
