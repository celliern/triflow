#!/usr/bin/env python
# coding=utf8


import numpy as np

from triflow import Model


# def test_routines_api():
#     x = np.linspace(0, 10, 100)
#     U = np.cos(x * 2 * np.pi / 10)
#     model = Model(evolution_equations="dxxU",
#                   dependent_variables="U")
#     fields = model.fields_template(x=x, U=U)
#     model.F(fields)
#     model.J(fields)
