#!/usr/bin/env python
# coding=utf8

import numpy as np

from triflow import Model


def test_fields_api():
    model = Model(differential_equations=["dxxU1", "dxxU2"],
                  dependent_variables=["U1", "U2"], help_functions='s')
    x = np.linspace(0, 1, 100)
    U1 = np.cos(x)
    U2 = np.sin(x)
    s = np.sin(x)
    fields = model.fields_template(x=x, U1=U1, U2=U2, s=s)
    print(fields)
    assert np.isclose(fields.uflat,
                      np.vstack([U1, U2]).flatten('F')).all()
    assert np.isclose(fields.flat,
                      np.vstack([x, U1, U2, s]).flatten('F')).all()
    for field, tested_field in zip(fields, np.vstack([x, U1, U2, s]).T):
        assert np.isclose(field, tested_field).all()
