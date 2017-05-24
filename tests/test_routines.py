#!/usr/bin/env python
# coding=utf8


import numpy as np

from triflow import Model


def test_routines_api():
    x = np.linspace(0, 10, 100)
    U = np.cos(x * 2 * np.pi / 10)
    model = Model(differential_equations="dxxU",
                  dependent_variables="U")
    fields = model.fields_template(x=x, U=U)
    F = model.F
    print(F)
    F(fields, dict(periodic=True))
    F.diff_approx(fields, dict(periodic=True))
    J = model.J
    print(J)
    J(fields, dict(periodic=True))
