#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
from triflow.core.model import Model


@pytest.fixture()
def fourrier_args():
    return ("-k * dx(dx(U))", "U", "k")


@pytest.fixture()
def model_monovariate_str(fourrier_args):
    func, var, par = fourrier_args
    model = Model(func=func,
                  vars=var,
                  pars=par)
    return model


@pytest.fixture()
def model_monovariate_list(fourrier_args):
    func, var, par = fourrier_args
    model = Model(func=[func],
                  vars=[var],
                  pars=[par])
    return model


@pytest.fixture()
def model_monovariate_tuple(fourrier_args):
    func, var, par = fourrier_args
    model = Model(func=(func, ),
                  vars=(var, ),
                  pars=(par, ))
    return model


def test_model_inline_string_monovariate_vars(model_monovariate_str):
    assert model_monovariate_str.vars == ('U', )


def test_model_inline_tuple_monovariate_pars(model_monovariate_str):
    assert model.pars == ('k', )
