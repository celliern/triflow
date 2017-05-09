
The convection diffusion equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import functools as ft
    import multiprocessing as mp
    import logging
    
    import numpy as np
    from scipy.signal import gaussian
    
    import pylab as pl
    
    from triflow import Model, Simulation, schemes, displays
    
    pl.style.use('seaborn-white')
    
    %matplotlib inline

The convection–diffusion equation is a combination of the diffusion and
convection (advection) equations, and describes physical phenomena where
particles, energy, or other physical quantities are transferred inside a
physical system due to two processes: diffusion and convection.
(`Wikipedia <https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation>`__)

The equation reads

.. math:: \partial_{t}U = k \partial_{xx} U - c \partial_{x} U

with

-  :math:`U` the physical quantities transferred (it could be a chemical
   species concentration, the temperature of a fluid...)
-  :math:`k` a diffusion convection
-  :math:`c` a velocity, which will be constant in our example.

.. code:: ipython3

    model = Model("k * dxxU - c * dxU",
                  "U", ["k", "c"])

We discretize our spatial domain. ``retstep=True`` ask to return the
spatial step. We want periodic condition, so ``endpoint=True`` exclude
the final node (which will be redondant with the first node, :math:`x=0`
and :math:`x=100` are merged)

.. code:: ipython3

    x, dx = np.linspace(0, 100, 500, retstep=True, endpoint=False)

We initialize with three gaussian pulses for the initial condition

.. code:: ipython3

    U = (np.roll(gaussian(x.size, 10), x.size // 5) +
         np.roll(gaussian(x.size, 10), -x.size // 5) -
         gaussian(x.size, 20))
    
    fields = model.fields_template(x=x, U=U)
    
    pl.figure(figsize=(15, 4))
    pl.plot(fields.x, fields.U)
    pl.xlim(0, fields.x.max())
    pl.show()



.. image:: advection_diffusion_files/advection_diffusion_7_0.png


We precise our parameters. The default scheme provide an automatic
time\_stepping. We set the periodic flag to True.

.. code:: ipython3

    parameters = dict(k=.2, c=10, periodic=True)

We initialize the simulation.

.. code:: ipython3

    t = 0
    simulation = Simulation(model, t, fields, parameters,
                            dt=.1, tmax=30)

We iterate on the simulation until the end.

.. code:: ipython3

    pl.figure(figsize=(15, 10))
    for i, (t, fields) in enumerate(simulation):
        if i % 2 == 0:
            pl.fill_between(fields.x, fields.U + .1 * (i + 1),
                            fields.U.min() - 1,
                            color='darkred', zorder=-2 * i, alpha=.7)
            pl.plot(fields.x, fields.U + .1 * (i + 1), 
                    color='white',
                    zorder=-(2 * i - 1))
        print(f"t: {t:g}".ljust(80), end='\r')
    pl.xlim(0, fields.x.max())
    pl.show()


.. parsed-literal::

    t: 29.9                                                                         


.. image:: advection_diffusion_files/advection_diffusion_13_1.png


