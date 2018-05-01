
.. code:: ipython3

    from triflow import Model, Simulation, display_fields
    import scipy.signal as spsig
    import numpy as np
    
    import pylab as pl
    
    import holoviews as hv
    
    pl.style.use("publication")
    %matplotlib inline

.. code:: ipython3

    model = Model("dx((h**3 + h**2) * dx(-sigma * dxxh + alpha * (1 / h**3 - e / h**4)))",
                  "h", ["sigma", "alpha", "e"])

.. code:: ipython3

    x = np.linspace(0, 10, 200)
    
    e = 1E-1
    
    h = (-(x - 5) ** 2 + 5 ** 2)
    h /= h.max()
    h -= 0.95
    h *= 10
    h[h < e] = e
    
    h = spsig.savgol_filter(h, 51, 8)
    
    h = spsig.gaussian(x.size, 20) + e
    pl.figure()
    pl.plot(x, h)
    pl.figure()
    
    y = np.linspace(.05, .5, 1000)
    pl.plot(y, 1/3 * e / y**3 - 1/2 / y**2)




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f32912d6710>]



.. parsed-literal::

    /home/nicolas/.cache/pypoetry/virtualenvs/triflow-py3.6/lib/python3.6/site-packages/matplotlib/font_manager.py:1328: UserWarning: findfont: Font family ['serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))



.. image:: dropplet_spread_files/dropplet_spread_2_2.png



.. image:: dropplet_spread_files/dropplet_spread_2_3.png


.. code:: ipython3

    def hook(t, fields, pars):
        fields['h'][0] = 1E-3
        fields['h'][-1] = 1E-3
        return fields, pars
    
    alpha = .05
    simul = Simulation(model,
                       fields=model.fields_template(x=x, h=h),
                       dt=.01, tol=1E-1, tmax=.5, #  hook=hook,
                       parameters=dict(periodic=False, alpha=alpha, sigma=10, e=e),
                       )
    simul.attach_container()
    display_fields(simul)




.. raw:: html

    <div id='6775e9fd-8859-4d80-905f-d491bae7fe38' style='display: table; margin: 0 auto;'>
        <div id="fig_6775e9fd-8859-4d80-905f-d491bae7fe38">
          
    <div class="bk-root">
        <div class="bk-plotdiv" id="ab432c8e-03f6-480b-b547-dd52b5b62f9d"></div>
    </div>
        </div>
        </div>



.. code:: ipython3

    for t, fields in simul:
        print(t, end='\r')


.. parsed-literal::

    0.590000000000000276

.. parsed-literal::

    /home/nicolas/Documents/03-projets/01-python/01-repositories/triflow/triflow/plugins/container.py:130: FutureWarning: casting an xarray.Dataset to a boolean will change in xarray v0.11 to only include data variables, not coordinates. Cast the Dataset.variables property instead to preserve existing behavior in a forwards compatible manner.
      if concatenated_fields and self.path:
    /home/nicolas/Documents/03-projets/01-python/01-repositories/triflow/triflow/plugins/container.py:130: FutureWarning: casting an xarray.Dataset to a boolean will change in xarray v0.11 to only include data variables, not coordinates. Cast the Dataset.variables property instead to preserve existing behavior in a forwards compatible manner.
      if concatenated_fields and self.path:


.. code:: ipython3

    hmap = hv.Dataset(simul.container.data.isel(t=slice(None, None, 15))).to(
        hv.Curve, "x", "h"
    )

.. code:: ipython3

    %%output backend="matplotlib" filename="droplet_out"
    %%opts Curve [show_grid=True, fig_size=12]
    
    hv.GridSpace(hmap)


.. parsed-literal::

    /home/nicolas/.cache/pypoetry/virtualenvs/triflow-py3.6/lib/python3.6/site-packages/matplotlib/font_manager.py:1328: UserWarning: findfont: Font family ['serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))
    /home/nicolas/.cache/pypoetry/virtualenvs/triflow-py3.6/lib/python3.6/site-packages/matplotlib/font_manager.py:1328: UserWarning: findfont: Font family ['serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))




.. raw:: html

    <div id='139854485193112' style='display: table; margin: 0 auto;'><img src='data:image/png;base64,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' style='max-width:100%; margin: auto; display: block; '/></div>


