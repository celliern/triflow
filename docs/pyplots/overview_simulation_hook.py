import matplotlib
import numpy as np

from triflow import Model, Simulation

matplotlib.use('Agg')  # noqa
import pylab as pl  # isort:skip

pl.style.use('seaborn-whitegrid')

model = Model("k * dxxU - c * dxU",
              "U", ["k", "c"])

x, dx = np.linspace(0, 1, 200, retstep=True)
U = np.cos(2 * np.pi * x * 5)
fields = model.fields_template(x=x, U=U)

parameters = dict(c=.03, k=.001, dx=dx, periodic=False)

t = 0
dt = 5E-1
tmax = 2.5

pl.plot(fields.x, fields.U, label=f't: {t:g}')


def dirichlet_condition(t, fields, pars):
    fields.U[0] = 1
    fields.U[-1] = 0
    return fields, pars


simul = Simulation(model, t, fields, parameters, dt,
                   hook=dirichlet_condition, tmax=tmax)

for i, (t, fields) in enumerate(simul):
    print(f"iteration: {i}\t",
          f"t: {t:g}", end='\r')
    pl.plot(fields.x, fields.U, label=f't: {t:g}')

pl.xlim(0, 1)
legend = pl.legend(loc='best')

pl.show()
