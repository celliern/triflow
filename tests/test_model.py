#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pytest

from triflow import Model


@pytest.mark.parametrize("func", np.array([[expr, [expr]]
                                           for expr
                                           in ["k * dxxU - c * dxU",
                                               "k * dx(dxU) - c * dx(U)"]],
                                          dtype=object)
                         .flatten().tolist())
@pytest.mark.parametrize("var", [func("U") for func in (str, tuple, list)])
@pytest.mark.parametrize("par", [func(["k", "c"]) for func in (tuple, list)])
@pytest.mark.parametrize("k", [1, np.ones((100,))])
def test_model_monovariate(func, var, par, k):
    model = Model(func,
                  var,
                  par)
    x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
    U = np.cos(x * 2 * np.pi / 10)
    fields = model.fields_template(x=x, U=U)
    parameters = dict(periodic=True, k=k, c=1E-3)
    F = model.F(fields, parameters)
    J_sparse = model.J(fields, parameters)
    J_dense = model.J(fields, parameters, sparse=False)
    J_approx = model.F.diff_approx(fields, parameters)

    dxU = np.gradient(np.pad(U, 2, mode='wrap')) / dx
    dxxU = np.gradient(dxU) / dx

    dxU = dxU[2:-2]
    dxxU = dxxU[2:-2]

    assert np.isclose(F, k * dxxU - 1E-3 * dxU, atol=1E-3).all()
    assert np.isclose(J_approx, J_sparse.todense(), atol=1E-3).all()
    assert np.isclose(J_approx, J_dense, atol=1E-3).all()


@pytest.mark.parametrize("func", [["k1 * dxx(v)",
                                   "k2 * dxx(u)"]])
@pytest.mark.parametrize("var", [func(["u", "v"])
                                 for func in (tuple, list)])
@pytest.mark.parametrize("par", [func(["k1", "k2"]) for func in (tuple, list)])
def test_model_bivariate(func, var, par):
    model = Model(func,
                  var,
                  par)
    x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
    u = np.cos(x * 2 * np.pi / 10)
    v = np.sin(x * 2 * np.pi / 10)
    fields = model.fields_template(x=x, u=u, v=v)
    parameters = dict(periodic=True, k1=1, k2=1)
    F = model.F(fields, parameters)
    J_sparse = model.J(fields, parameters)
    J_dense = model.J(fields, parameters, sparse=False)
    J_approx = model.F.diff_approx(fields, parameters)

    dxu = np.gradient(np.pad(u, 2, mode='wrap')) / dx
    dxxu = np.gradient(dxu) / dx
    dxu = dxu[2:-2]
    dxxu = dxxu[2:-2]

    dxv = np.gradient(np.pad(v, 2, mode='wrap')) / dx
    dxxv = np.gradient(dxv) / dx
    dxv = dxv[2:-2]
    dxxv = dxxv[2:-2]
    assert np.isclose(F, np.vstack([dxxv, dxxu]).flatten('F'),
                      atol=1E-3).all()
    assert np.isclose(J_approx, J_sparse.todense(), atol=1E-3).all()
    assert np.isclose(J_approx, J_dense, atol=1E-3).all()
