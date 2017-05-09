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
    x, dx = np.linspace(0, 10, 500, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    t0 = 0
    for i, (t, fields) in enumerate(Simulation(heat_model, t0,
                                               initial_fields, parameters,
                                               scheme=scheme,
                                               dt=1, tmax=100)):
        continue
    assert t == 100
    assert np.isclose(fields.T.mean(), 0)


@pytest.mark.parametrize("scheme",
                         [schemes.ROS3PRL, schemes.ROS3PRw, schemes.RODASPR])
def test_simul_heat_eq_dirichlet(heat_model, scheme):
    x, dx = np.linspace(0, 10, 500, retstep=True, endpoint=False)
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
                       dt=.1, tmax=100, tol=1E-2)

    for i, (t, fields) in enumerate(simul):
        continue
    assert np.isclose(t, 100)
    assert np.isclose(fields.T, 1, atol=1E-2).all()
