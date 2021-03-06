import numpy as np
import pylab as pl

from triflow import Model, Simulation

model = Model("k * dxxU - c * dxU",
              "U", ["k", "c"])

x, dx = np.linspace(0, 1, 200, retstep=True)
U = np.cos(2 * np.pi * x * 5)
fields = model.fields_template(x=x, U=U)

parameters = dict(c=.03, k=.001, dx=dx, periodic=False)

t = 0
dt = 5E-1
tmax = 2.5

pl.plot(fields.x, fields.U, label='t: %g' % t)


def dirichlet_condition(t, fields, pars):
    fields.U[0] = 1
    fields.U[-1] = 0
    return fields, pars


simul = Simulation(model, fields, parameters, dt,
                   hook=dirichlet_condition, tmax=tmax)

for i, (t, fields) in enumerate(simul):
    print("iteration: %i\t" % i,
          "t: %g" % t, end='\r')
    pl.plot(fields.x, fields.U, label='t: %g' % t)

pl.xlim(0, 1)
legend = pl.legend(loc='best')

pl.show()
