#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from triflow import Model, Simulation
from triflow.plugins import schemes


@pytest.fixture
def heat_model():
    model = Model(differential_equations="k * dxxT",
                  dependent_variables="T",
                  parameters="k")
    return model


@pytest.mark.parametrize("scheme",
                         [schemes.ROS2, schemes.ROS3PRL, schemes.ROS3PRw,
                          schemes.RODASPR, schemes.Theta, schemes.scipy_ode])
def test_simul_heat_eq(heat_model, scheme):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    t0 = 0
    for i, (t, fields) in enumerate(Simulation(heat_model, t0,
                                               initial_fields, parameters,
                                               scheme=scheme,
                                               dt=1, tmax=100, tol=1E-1)):
        continue
    assert t == 100
    assert np.isclose(fields.T.mean(), 0)


@pytest.mark.parametrize("scheme",
                         [schemes.ROS3PRL, schemes.ROS3PRw, schemes.RODASPR])
def test_simul_heat_eq_dirichlet(heat_model, scheme):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=False, k=1)
    t0 = 0

    def dirichlet_bdc(t, fields, parameters):
        fields.T[0] = 1
        fields.T[-1] = 1
        return fields, parameters

    simul = Simulation(heat_model, t0, initial_fields,
                       parameters, hook=dirichlet_bdc,
                       scheme=scheme,
                       dt=.1, tmax=100, tol=1E-1)

    for i, (t, fields) in enumerate(simul):
        pass
    assert np.isclose(t, 100)
    assert np.isclose(fields.T, 1, atol=1E-1).all()


def test_simulation_api(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    t0 = 0

    class null_display():
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, t, fields):
            pass

    simul = Simulation(heat_model, t0,
                       initial_fields, parameters,
                       dt=1, tol=1E-1)
    simul.add_display(null_display)

    for i, (t, fields) in enumerate(simul):
        if i > 10:
            break
    next(simul)


def test_runtime_error(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    t0 = 0

    simul = Simulation(heat_model, t0,
                       initial_fields, parameters,
                       dt=1, tol=1E-1, max_iter=2)
    with pytest.raises(RuntimeError):
        for t, fields in simul:
            pass

    simul = Simulation(heat_model, t0,
                       initial_fields, parameters,
                       dt=1, tol=1E-1, dt_min=.1)
    with pytest.raises(RuntimeError):
        for t, fields in simul:
            pass
