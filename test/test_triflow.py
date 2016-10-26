#!/usr/bin/env python
# coding=utf8

import pytest
from triflow.make_routines import make_routines_fortran, generate_functions
from triflow.model_4fields import model
import numpy as np


@pytest.fixture()
def generate_flat_film():
    # U, F, J = model()
    # get_fields, flatten_fields, compute_F = generate_functions(model)
    # %% Use it

    L = 200
    Nxi = 200
    x, dxi = np.linspace(0, L, Nxi, endpoint=False, retstep=True)

    Ka = 3000
    Re = 15  # (7.5)#/0.8**3
    We = Ka / ((3 * Re)**(2 / 3))
    Bit = 1
    Pr = 30

    Pe = Pr * Re
    Bi = (2 * Re)**(1 / 3) * Bit
    M = 0
    Ct = 0

    h = x * 0 + 1
    q = x * 0 + .33333
    theta = .5 + x * 0
    phi = x * 0 - .03
    s = x * 0

    parameters = {'Re': Re, 'Pe': Pe, 'We': We,
                  'Ct': Ct, 'M': M, 'Bi': Bi, 'Delta_x': dxi}
    return (h, q, theta, phi, s), parameters


def test_get(generate_flat_film):
    get_fields, flatten_fields, compute_F = generate_functions(model)
    (h, q, theta, phi, s), parameters = generate_flat_film

    return (np.array(get_fields(flatten_fields(h, q, theta, phi, s))).all() ==
                np.array([h, q, theta, phi, s]).all())
