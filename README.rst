Installation
===============


External requirements
---------------------

The library is based on Theano, thus extra dependecies like fortran and C compiler are needed, see Theano install page for extra informations:

http://deeplearning.net/software/theano/install.html


via PyPI
---------

Beware, the PyPI version is not always up-to-date.

.. code:: bash

    pip install triflow

will install the package and

.. code:: bash

    pip install triflow --upgrade

will update an old version of the library.

use sudo if needed, and the user flag if you want to install it without the root privileges:

.. code:: bash

    pip install --user triflow

.. via Conda
.. ----------

.. The library is also available on a conda channel (not always up to date) :

.. .. code:: bash

..     conda install -c celliern triflow


via github
-----------

You can install the last version of the library using pip and the github repository:

.. code:: bash

    pip install git+git://github.com/locie/triflow.git


Introduction
===============

Motivation
-----------------

The aim of this library is to have a (relatively) easy way to write transient dynamic systems with 1D finite difference discretisation, with fast temporal solvers.

The main two parts of the library are:
* symbolic tools defining the spatial discretisation, with boundary taking into account in a separated part
* a fast temporal solver written in order to use the sparsity of the finite difference method to reduce the memory and CPU usage during the solving

Moreover, extra tools are provided and the library is written in a modular way, allowing an easy extension of these different parts (see the plug-in module of the library.)

The library fits well with an interactive usage (in a jupyter notebook). The dependency list is actually larger, but on-going work target a reduction of the stack complexity.

Model writing
-----------------

All the models are written as function generating the F vector and the Jacobian matrix of the model defined as

.. math::

    \frac{\partial U}{\partial t} = F(U)

The symbolic model is written as a simple mathematic equation. For exemple, a diffusion advection model can be written as:

.. code-block:: python

    from triflow import Model

    func = "k * dxxU - c * dxU"
    var = "U"
    pars = ["k", "c"]

    model = Model(func, var, pars)

Example
-------


.. code-block:: python

    import numpy as np
    from triflow import Model, Simulation
    from triflow.plugins.displays import bokeh_probes_update


    model = Model(funcs="k * dxxU - c * dxU", vars="U", pars=["k", "c"])
    parameters = dict(time_stepping=True,
                      tol=1E-1, dt=1, tmax=100,
                      periodic=True,
                      c=1, k=1E-6)

    x = np.linspace(-2 * np.pi, 2 * np.pi, 100, endpoint=False)
    U = np.cos(x) + 2

    fields = model.fields_template(x=x, U=U)
    simul = Simulation(model, fields, 0, parameters)

    def internal_iter(t, simul):
        return simul.scheme.internal_iter

    bokeh_probe = bokeh_probes_update({'niter': internal_iter})

    for fields, t in simul:
        bokeh_probe.send((t, simul))

.. _Theano: http://deeplearning.net/software/theano/
.. _Sympy: http://www.sympy.org/en/index.html
.. _Numpy: http://www.sympy.org/en/index.html
.. _scipy sparse column matrix format: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csc_matrix.html
.. _SuperLU: http://crd-legacy.lbl.gov/~xiaoye/SuperLU/
