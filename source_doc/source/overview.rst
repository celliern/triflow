Overview
===============

Motivation
-----------------

The aim of this library is to have an easy way to write
transient dynamic systems with 1D finite
difference discretisation, with fast temporal solvers.

The main two parts of the library are:

* symbolic tools defining the spatial discretisation.
* a fast temporal solver written to use the sparsity
    of the finite difference method to reduce the memory
    and CPU usage during the computation. Theano_ make this part easy.

Moreover, we provide extra tools and we write the library in a
modular way, allowing an easy extension of these different
parts (see the plug-in module of the library.)

The library fits well with an interactive usage (in a jupyter notebook).

This is just an overview : more details are in thededicated pages of
each submodules.

Model writing
-----------------

We write all the models as function generating the F vector and the
Jacobian matrix of the model defined as

.. math::

    \frac{\partial U}{\partial t} = F(U)

We write the symbolic model as a simple mathematic equation. For exemple,
a diffusion advection model:

.. code-block:: python3

    >>> import triflow as trf

    >>> eq_diff = "k * dxxU - c * dxU"
    >>> dep_var = "U"
    >>> pars = ["k", "c"]

    >>> model = trf.Model(eq_diff, dep_var, pars)

the model give us access after that to the compiled routines for F and
the corresponding Jacobian matrix as:

.. code-block:: python3

    >>> print(model.F)
    Matrix([[-2*U*k/dx**2 + 0.5*U_m1*c/dx + U_m1*k/dx**2 - 0.5*U_p1*c/dx + U_p1*k/dx**2]])

    >>> print(model.J)
    Matrix([
    [ 0.5*c/dx + k/dx**2],
    [         -2*k/dx**2],
    [-0.5*c/dx + k/dx**2]])

We compute the Jacobian in a sparse form. These object are
callable, and will return the numerical values if we provide
the fields and the parameters:

.. code-block:: python3

    >>> print(model.F(initial_fields, parameters))  # doctest: +ELLIPSIS
    [...]

    >>> print(model.J(initial_fields, parameters))  # doctest: +SKIP
    <NxN sparse matrix of type '<class 'numpy.float64'>'
    with M stored elements in Compressed Sparse Column format>

a numerical approximation is available for debug purpose with

.. code-block:: python3

    >>> print(model.F(initial_fields, parameters))
    [...]

be aware that numerical approximation of the Jacobian is far less
efficient than the provided optimized routine.

optional arguments : helpers function and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model take two mandatory parameters: differential_equations,
dependent_variables. The first define the evaluation of the time derivative,
the second the name of the dependant variables.

It can take two optional arguments :


* parameters, a list of parameters name. They can be scalar or vector with the
    same dimension as the dependant variables.
* help_functions, a list of outside variables : they have to be vector
    with the same dimension of the dependant variable.

So, what is main difference between them? The difference is that you
have the possibility to use spatial derivative of the fields in the model.
Because the fields are parsed and the derivative approximated,
it make the graph optimization of the model grows.


Model compilation
------------------

The model has to be compiled before being employed. The sympy_ library
provides an easy way to automatically write the Fortran or C routine
corresponding. Better than that, the symbolic form of the expression feed
custom compilers in order to provide the routine for the time derivative
and the associate jacobian.

Actually there are two different compilers : the first one use only the
NumPy_ library (and is not really compiled, but use NumPy_ mechanism
which stand by C array operations). In that case the initialization time
depend only of the symbolic computation and can be almost instant for
simple models. The second one is build on Theano_, thus provide algorithm
graph optimization and write a C binary for the routines. For simple case,
the Theano_ compiler is twice faster as the NumPy_ one.
By default, Theano_ is used.

In the examples folder live some classic 1D PDE
(diffusion, diffusion/advection, burger equation...).

The Model class is pickable, means that it can be sent across
the network and between cpu for multiprocessing purpose.
It can be save on disk as a binary and reload later.
It is important in order to reduce the large compilation overhead.
(see Model.save and load_model). Thus, the model has to be
re-optimized by Theano on every new host if using this compiler,
leading to potential long initialization for large and complex models.
The memory footprint can be large (> 1Go) in some case: this is the
cost of the theano aggressive graph optimization strategy.
It should be important to notice that Theano_ is able to handle
GPU computation if properly configured
(see its documentation for more details). For large parametric
studies and simple models, using the NumPy_
compiler could be more interesting.

Fields containers
------------------

A special container has been designed to handle
initial values of the dependant solutions (the unknowns),
the independant variables (spatial coordinates),
the constant fields and the post-processed variable
(known as helper function).

A factory is linked to the model and is accessible via
the model.fields_template property :

.. code-block:: python3

    >>> import numpy as np
    >>> import triflow as trf

    >>> model = trf.Model("k * dxxU - c * dxU",
    ...                   "U", ["k", "c"])

    >>> x, dx = np.linspace(0, 1, 100, retstep=True)
    >>> U = np.cos(2 * np.pi * x * 5)
    >>> initial_fields = model.fields_template(x=x, U=U)

The variable involved in the computation are stored on a large
vector containing all the fields, and this object give access
to each fields to simplify their modification and the computations.
This container is built on the top of the xarray_ library, a pandas-like data container
for multi-dimensional sets. Triflow BaseFields derived from
the xarray Dataset_ thus have the same API for most methods and attributes.

.. code-block:: python3

    >>> initial_fields.U[:] = 5
    >>> print(initial_fields.U)
    <xarray.DataArray 'U' (x: ...)>
    array([5., 5., ..., 5., 5.])
    Coordinates:
      * x        (x) float64 ...

Numerical scheme, temporal solver
----------------------------------

In order to provide fast and scalable temporal solver, the
Jacobian use the `scipy sparse column matrix format`_
(which will reduce the memory usage, especialy for a large
number of spatial nodes), and make available the SuperLU_
decomposition, a fast LU sparse matrix decomposition algorithm.

Different temporal schemes are provided in the plugins module:

* a forward Euler scheme
* a backward Euler scheme
* a :math:`\theta` mixed scheme
* A ROW schemes from order 3 up to 6 with fixed and variable time stepping.
* A proxy schemes giving access to all the scipy.integrate.ode schemes.

Each of them have advantages and disadvantages.

They can accept somme extra arguments during their instantiation
(for exemple the :math:`\theta` parameter for the :math:`\theta` mixed scheme),
and are called with the actual fields, time, time-step, parameters,
and accept an optionnal hook modifying fields and parameters each
time the solver compute the function or its jacobian.

The following code compute juste one time-step with a Crank-Nicolson scheme.

.. plot:: pyplots/overview_model_one_step.py
   :include-source:

We obtain with the following code a full resolution up to the target time.

.. plot:: pyplots/overview_model_multi_step.py
   :include-source:

hook and boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hook function is used in order to deal with variable and
conditional parameters and boundary condition.

Inside the model, the fields are padded in order to solve the
equation. If the parameter "periodic" is used, the pad function
is used with the mode "wrap" leading to periodic fields.
If not, the mode "edge" is used, repeating the first and
last node. It is very easy to implement Dirichlet condition
with the following function:

.. plot:: pyplots/overview_model_hook.py
   :include-source:

The hook function is used in order to deal with variable and
conditional parameters and boundary condition.

Inside the model, the fields are padded in order to solve the
equation. If the parameter "periodic" is used, the pad function is used with
the mode "wrap" leading to periodic fields. If not, the mode "edge" is used,
repeating the first and last node. It is very easy to implement Dirichlet
condition with the following function:

.. plot:: pyplots/overview_model_hook.py
   :include-source:

Time Stepping
^^^^^^^^^^^^^

Some of the provided solvers come with built-in automatic time stepping.
For the others, an external tool is provided in the same module as the
temporal schemes. It takes a scheme as input and return a decorated scheme
with internal time-stepping. This one is not as efficient as the built-in
(and require extra computation) but is handy to deal with simulation that
has varying kinetics. It is used by default in the Simulation Class
(see bellow).

Simulation class: higher level control
--------------------------------------

The loop snippet

.. code-block:: python3

    >>> import itertools as it
    >>> scheme = trf.schemes.RODASPR(model)
    >>> t = t0
    >>> fields = initial_fields.copy()
    >>> for i in it.count():  # doctest: +SKIP
    ...     t, fields = scheme(t, fields, dt, parameters)
    ...     if t > tmax:
    ...         break

is not handy.

To avoid it, we provide a higher level control class, the Simulation.
It is an iterable and we can write the snippet as:

.. code-block:: python3

    >>> simul = trf.Simulation(model, initial_fields, parameters, dt, tmax=tmax)
    >>> simul.run()  # doctest: +SKIP

and we write the previous advection-diffusion example as:

.. plot:: pyplots/overview_simulation_hook.py
   :include-source:

Post-processing
^^^^^^^^^^^^^^^

It is possible to add one or more post-process to the simulation. They will be
called juste after the simulation step : if extra variables are add to
the fields, they will be accessible for display and be saved on disk if a
container is attached to the simulation.


Container
^^^^^^^^^

Containers are special data structure meant to keep the simulation history.
They are in-memory by default, but can be persistent if a path is provided by
the user.

The two main attributes are :python3: `container.data` and
:python3: `container.metadata`, which contains the content of the simulation
fields for each timestep and the simulation parameters.

.. code-block:: python3

    >>> simul = trf.Simulation(model, initial_fields, parameters, dt, tmax=tmax)
    >>> simul.attach_container()  # doctest: +SKIP
    >>> simul.run()  # doctest: +SKIP
    >>> simul.container  # doctest: +SKIP
    path:   None
    <xarray.Dataset>
    Dimensions:  (t: 502, x: 100)
    Coordinates:
    * x        (x) float64 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 ...
    * t        (t) float64 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 ...
    Data variables:
        U        (t, x) float64 1.0 0.9511 0.809 0.5878 0.309 6.123e-17 -0.309 ...
    Attributes:
        c:         0.03
        k:         0.001
        periodic:  1

If a persistent container is requested, the data live in
:python3: `path / simulation.id / "data.nc"` and the metadata in
:python3: `path / simulation.id / "metadata.yml"`. They can be easily imported
with :python3: `trf.retrieve_container("path/to/container/folder")`.

.. code-block:: python3

    >>> simul = trf.Simulation(model, initial_fields, parameters, dt, tmax=tmax)
    >>> simul.attach_container(my_directory)  # doctest: +SKIP
    >>> simul.run()  # doctest: +SKIP
    >>> trf.retrieve_container("%s/%s" % (my_directory, simul.id))  # doctest: +SKIP


Displays
^^^^^^^^

Triflow allows real-time display of the simulations. It rely on Holoviews_ with
Bokeh_ if the user is within a jupyter lab or notebook, or  matplotlib_ if
it run on a non-interactive way.

For an interactive usage, see `this beautiful example`_.

If it's for a non-interactive usage, triflow can save the plots on disk for
each timestep.

.. code-block:: python3

    >>> simul = trf.Simulation(model, initial_fields, parameters, dt, tmax=tmax)
    >>> trf.display_fields(simul, on_disk="plot/output/")
    >>> simul.run()  # doctest: +SKIP


Post-processing
^^^^^^^^^^^^^^^

It is possible to add one or more post-process to the simulation. They will be
called juste after the simulation step : if extra variables are add to
the fields, they will be accessible for display and be saved on disk if a
container is attached to the simulation.

.. code-block:: python3

    >>> simul = trf.Simulation(model, initial_fields, parameters, dt, tmax=tmax)
    >>> def compute_gradient(simul):
    ...     simul.fields["grad"] = "x", (np.gradient(simul.fields["U"]) /
                                        np.gradient(simul.fields["x"]))
    >>> simul.add_post_process("grad", compute_gradient)
    >>> simul.attach_container()
    >>> trf.display_fields(simul, "grad")
    >>> simul.run()  # doctest: +SKIP
    >>> simul.container.data["grad"]  # doctest: +SKIP

    <xarray.DataArray 'grad' (t: 21, x: 200)>
    array([[-2.462332, -4.894348, -9.668182, ..., 14.203952,  9.668182,  7.326365],
        [ 0.435707, -1.791205, -6.19556 , ..., 15.182621, 11.249958,  9.201766],
        [ 2.808271,  0.787149, -3.239902, ..., 15.731606, 12.360189, 10.587481],
        ...,
        [ 5.326002,  5.309953,  5.212679, ...,  4.415704,  4.823781,  4.999205],
        [ 4.787014,  4.808028,  4.790602, ...,  3.732835,  4.162767,  4.35328 ],
        [ 4.264091,  4.315188,  4.363626, ...,  3.109441,  3.550107,  3.749816]])
    Coordinates:
    * x        (x) float64 0.0 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 ...
    * t        (t) float64 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 ...

.. _Theano: http://deeplearning.net/software/theano/
.. _Sympy: http://www.sympy.org/en/index.html
.. _NumPy: http://www.sympy.org/en/index.html
.. _Bokeh: https://bokeh.pydata.org/en/latest/
.. _Holoviews: https://holoviews.org/
.. _`this beautiful example`: https://
.. _matplotlib: https://matplotlib.org/
.. _xarray: http://xarray.pydata.org/en/stable/
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html?highlight=dataset
.. _scipy sparse column matrix format: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csc_matrix.html
.. _SuperLU: http://crd-legacy.lbl.gov/~xiaoye/SuperLU/
