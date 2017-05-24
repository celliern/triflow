#!/usr/bin/env python
# coding=utf8

import numpy as np

from triflow import Model, Simulation

# We initialize the model dtU = k * dxxU and we precise
# the variable and the parameters
model = Model("k * dxxU",
              "U", "k")

# We discretize our spatial domain between 0 and 100 with 500 nodes.
# retstep=True ask to return the spatial step. We want periodic condition,
# so endpoint=True exclude the final node (which will be redondant with the
# first node, x=0 and x=100 are merged)
x, dx = np.linspace(0, 100, 500, retstep=True, endpoint=False)

# We initialize with a sinusoidal initial condition
U = np.cos(2 * np.pi * x / 100 * 10)
# We fill the fields container
fields = model.fields_template(x=x, U=U)
# We precise our parameters. The default scheme provide an automatic
# time_stepping, we have to precise the tolerance. We set a periodic
# simulation.
parameters = dict(k=1e-1, periodic=True)


# We initialize the simulation
t = 0
simulation = Simulation(model, t, fields, parameters, dt=5, tol=1E-1, tmax=50)

# We iterate on the simulation until the end.
for t, fields in simulation:
    print(f"t: {t:g}, mean value of U: {np.mean(fields.U):g}")
