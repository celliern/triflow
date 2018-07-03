#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from path import tempdir

from triflow import Model


@pytest.fixture
def heat_model():
    model = Model(
        evolution_equations="k * dxxT",
        dependent_variables="T",
        parameters="k",
        boundary_conditions={"U": {
            "x": "periodic"
        }})
    return model


def test_model_monovariate():
    model = Model("k * dxxU", "U", "k(x)", boundary_conditions={"U": {"x": "periodic"}})
    x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
    U = np.cos(x * 2 * np.pi / 10)
    k = np.ones_like(U)
    fields = model.fields_template(x=x, U=U, k=k)
    F = model.F(fields)
    J_sparse = model.J(fields)

    dxU = np.gradient(np.pad(U, 2, mode='wrap')) / dx
    dxxU = np.gradient(dxU) / dx

    dxU = dxU[2:-2]
    dxxU = dxxU[2:-2]

    assert np.isclose(F, k * dxxU, atol=1E-3).all()


def test_model_bivariate():
    model = Model(
        ["k1 * dxx(v)", "k2 * dxx(u)"], ["u", "v"], ["k1", "k2"],
        boundary_conditions={
            "u": {
                "x": "periodic"
            },
            "v": {
                "x": "periodic"
            }
        })
    x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
    u = np.cos(x * 2 * np.pi / 10)
    v = np.sin(x * 2 * np.pi / 10)
    fields = model.fields_template(x=x, u=u, v=v, k1=1, k2=1)
    F = model.F(fields)

    dxu = np.gradient(np.pad(u, 2, mode='wrap')) / dx
    dxxu = np.gradient(dxu) / dx
    dxu = dxu[2:-2]
    dxxu = dxxu[2:-2]

    dxv = np.gradient(np.pad(v, 2, mode='wrap')) / dx
    dxxv = np.gradient(dxv) / dx
    dxv = dxv[2:-2]
    dxxv = dxxv[2:-2]
    assert np.isclose(F, np.vstack([dxxv, dxxu]).flatten('F'), atol=1E-3).all()


@pytest.mark.parametrize(
    "args", [('dxU', lambda x: -np.sin(x)), ('dxxU', lambda x: -np.cos(x)),
             ('dxxxU', lambda x: np.sin(x)), ('dxxxxU', lambda x: np.cos(x))])
def test_finite_diff(args):
    symb_diff, analytical_func = args
    model = Model(symb_diff, 'U', boundary_conditions={"U": {"x": "periodic"}})
    x = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    triflow_diff = model.F(
        model.fields_template(x=x, U=np.cos(x)), dict(periodic=True))
    assert (np.isclose(triflow_diff, analytical_func(x), rtol=1E-4,
                       atol=1E-4).all())


# def test_model_api():
#     model = Model(
#         evolution_equations=["k * dxxU + s"],
#         dependent_variables="U",
#         parameters=['k', 's(x)'],)
#     x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
#     U = np.cos(x * 2 * np.pi / 10)
#     s = np.zeros_like(x)
#     fields = model.fields_template(x=x, U=U, s=s, k=1)
#     model.F(fields)
#     model.J(fields)


# @pytest.mark.parametrize("uporder", [1, 2, 3])
# @pytest.mark.parametrize("vel", ["1", "U"])
# def test_upwind(compiler, uporder, vel, periodic):
#     model = Model(
#         evolution_equations=["upwind(%s, U, %i)" % (vel, uporder)],
#         dependent_variables="U",
#         parameters='k',
#         help_functions='s')

#     x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
#     U = np.cos(x * 2 * np.pi / 10)
#     s = np.zeros_like(x)
#     fields = model.fields_template(x=x, U=U, s=s)
#     parameters = dict(periodic=periodic, k=1)
#     model.F(fields, parameters)
#     model.J(fields, parameters)
