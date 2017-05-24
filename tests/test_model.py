#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pytest
from path import tempdir

from triflow import Model


@pytest.fixture
def heat_model():
    model = Model(differential_equations="k * dxxT",
                  dependent_variables="T",
                  parameters="k")
    return model


@pytest.mark.parametrize("func", np.array([[expr, [expr]]
                                           for expr
                                           in ["k * dxxU",
                                               "k * dx(dxU)"]],
                                          dtype=object)
                         .flatten().tolist())
@pytest.mark.parametrize("var", [func("U") for func in (str, tuple, list)])
@pytest.mark.parametrize("par", [func("k") for func in (str, tuple, list)])
@pytest.mark.parametrize("k", [1, np.ones((100,))])
def test_model_monovariate(func, var, par, k):
    model = Model(func,
                  var,
                  par)
    x, dx = np.linspace(0, 10, 100, retstep=True, endpoint=False)
    U = np.cos(x * 2 * np.pi / 10)
    fields = model.fields_template(x=x, U=U)
    parameters = dict(periodic=True, k=k)
    F = model.F(fields, parameters)
    J_sparse = model.J(fields, parameters)
    J_dense = model.J(fields, parameters, sparse=False)
    J_approx = model.F.diff_approx(fields, parameters)

    dxU = np.gradient(np.pad(U, 2, mode='wrap')) / dx
    dxxU = np.gradient(dxU) / dx

    dxU = dxU[2:-2]
    dxxU = dxxU[2:-2]

    assert np.isclose(F, k * dxxU, atol=1E-3).all()
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


@pytest.mark.parametrize("args", [('dxU', lambda x: -np.sin(x)),
                                  ('dxxU', lambda x: -np.cos(x)),
                                  ('dxxxU', lambda x: np.sin(x)),
                                  ('dxxxxU', lambda x: np.cos(x))])
def test_finite_diff(args):
    symb_diff, analytical_func = args
    model = Model(symb_diff, 'U')
    x = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    triflow_diff = model.F(model.fields_template(x=x, U=np.cos(x)),
                           dict(periodic=True))
    assert (np.isclose(triflow_diff, analytical_func(x),
                       rtol=1E-4, atol=1E-4).all())


def test_model_api():
    model = Model(differential_equations=["k * dxxU + s"],
                  dependent_variables="U",
                  parameters='k', help_functions='s')
    assert set(model._args) == set(['x', 'U_m1', 'U', 'U_p1',
                                    's_m1', 's', 's_p1',
                                    'k', 'dx'])
    with pytest.raises(NotImplementedError):
        Model('dxxxxxU', 'U')
    with pytest.raises(ValueError):
        Model('dxxx(dx)', 'U')


def test_save_load(heat_model):

    with tempdir() as d:
        heat_model.save(d / 'heat_model')
        loaded_heat_model = Model.load(d / 'heat_model')

        x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
        T = np.cos(x * 2 * np.pi / 10)
        initial_fields = heat_model.fields_template(x=x, T=T)
        parameters = dict(periodic=True, k=1)

        assert (loaded_heat_model._symb_diff_eqs ==
                heat_model._symb_diff_eqs)
        assert (loaded_heat_model._symb_dep_vars ==
                heat_model._symb_dep_vars)
        assert (loaded_heat_model._symb_pars ==
                heat_model._symb_pars)
        assert (loaded_heat_model._symb_help_funcs ==
                heat_model._symb_help_funcs)
        assert (loaded_heat_model.F_array ==
                heat_model.F_array)
        assert (loaded_heat_model.J_array ==
                heat_model.J_array).all()
        assert (loaded_heat_model._J_sparse_array ==
                heat_model._J_sparse_array).all()
        assert (list(map(str, loaded_heat_model._th_args)) ==
                list(map(str, heat_model._th_args)))
        assert (list(map(str, loaded_heat_model._th_vectors)) ==
                list(map(str, heat_model._th_vectors)))
        assert (loaded_heat_model.F(initial_fields, parameters) ==
                heat_model.F(initial_fields, parameters)).all()
        assert (loaded_heat_model.J(initial_fields, parameters).todense() ==
                heat_model.J(initial_fields, parameters).todense()).all()
