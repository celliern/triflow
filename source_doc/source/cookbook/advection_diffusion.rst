
The convection diffusion equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import numpy as np
    import pylab as pl
    import triflow as trf
    from scipy.signal import gaussian
    
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
   species concentration, the temperature of a fluid…)
-  :math:`k` a diffusion convection
-  :math:`c` a velocity, which will be constant in our example.

.. code:: ipython3

    model = trf.Model("k * dxxU - c * dxU", "U", ["k", "c"])

We discretize our spatial domain. We want periodic condition, so
``endpoint=True`` exclude the final node (which will be redondant with
the first node, :math:`x=0` and :math:`x=100` are merged)

.. code:: ipython3

    x = np.linspace(0, 100, 500, endpoint=False)

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
time_stepping. We set the periodic flag to True.

.. code:: ipython3

    parameters = dict(k=.2, c=10, periodic=True)

We initialize the simulation.

.. code:: ipython3

    %%opts Curve [show_grid=True, width=800] {-framewise}
    simulation = trf.Simulation(model, fields, parameters, dt=.1, tmax=30)
    container = simulation.attach_container()
    trf.display_fields(simulation)




.. raw:: html

    <div id='dc7acc10-0494-4d4e-a422-57b7bea23bc0' style='display: table; margin: 0 auto;'>
        <div id="fig_dc7acc10-0494-4d4e-a422-57b7bea23bc0">
          
    <div class="bk-root">
        <div class="bk-plotdiv" id="f77e8042-eacd-4668-acc4-35f0db12971b"></div>
    </div>
        </div>
        </div>



We iterate on the simulation until the end.

.. code:: ipython3

    result = simulation.run()



.. parsed-literal::

    HBox(children=(IntProgress(value=0, max=300), HTML(value='')))


.. parsed-literal::

    


.. code:: ipython3

    container.data.U.plot()




.. parsed-literal::

    <matplotlib.collections.QuadMesh at 0x7f349d36ac88>




.. image:: advection_diffusion_files/advection_diffusion_14_1.png

