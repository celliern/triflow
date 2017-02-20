#!/usr/bin/env python
# coding=utf8

import numpy as np
from triflow import Model, Simulation
import pylab as pl

# We initialize the model dtU = k * dxxU - c * dxU and we precise
# the variable and the parameters
model = Model("k * dxxU + c * dxU",
              "U", ["k", "c"])

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
parameters = dict(k=1e-1, c=40, dx=dx, time_stepping=True,
                  tol=1E-1, periodic=True, tmax=50, dt=5)


# We initialize the simulation
t = 0
simulation = Simulation(model, fields, t, parameters)

# We plot the initial condition
pl.plot(fields.x, fields.U, color='black', linestyle='--')

# We iterate on the simulation until the end.
for fields, t in simulation:
    # We plot each time step
    pl.plot(fields.x, fields.U, alpha=t / 200, color='black')
    print(f"t: {t:g}, mean value of U: {np.mean(fields.U):g}")

# We plot the final solution
pl.plot(fields.x, fields.U, color='black')
pl.show()
