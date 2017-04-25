#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest

import numpy as np
from triflow import Model


@pytest.mark.parametrize("func", np.array([[expr, (expr, ), [expr]]
                                           for expr
                                           in ["-k * dxxU + dxU",
                                               "-k * dx(U, 2) + dx(U, 1)",
                                               "-k * dxx(U) + dx(U)"]],
                                          dtype=object)
                         .flatten().tolist())
@pytest.mark.parametrize("var", [func("U") for func in (str, tuple, list)])
@pytest.mark.parametrize("par", [func("k") for func in (str, tuple, list)])
def test_model_monovariate(func, var, par):
    model = Model(func,
                  var,
                  par)
    return model


@pytest.mark.parametrize("func", [["k1 * dx(v, 2)",
                                   "k2 * dx(u, 2)"],
                                  {"u": "k1 * dx(v, 2)",
                                   "v": "k2 * dx(u, 2)"}])
@pytest.mark.parametrize("var", [func(["u", "v"])
                                 for func in (tuple, list)])
@pytest.mark.parametrize("par", [func(["k1", "k2"]) for func in (tuple, list)])
def test_model_bivariate(func, var, par):
    model = Model(func,
                  var,
                  par)
    return model
