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

This part will have a huge arrangement in the future: do not trust the ape!
For now, all the models are written as function generating the F vector and the Jacobian matrix of the model defined as

.. math::

    \frac{\partial U}{\partial t} = F(U)

The symbolic model is written as a simple mathematic equation. For exemple, a diffusion advection can be written as:

.. code-block:: python

    from triflow import Model

    func = "k * dxxU - c * dxU"
    var = "U"
    pars = ["k", "c"]

    model = Model(func, var, pars)

Model compilation
------------------

The model has to be compiled before being employed. The sympy library provides an easy way to automatically write the Fortran or C routine corresponding. Better than that, a tool has been written in order to convert sympy complex expressions to Theano_ graph which can be easily compiled.

In the examples folder live some classic 1D PDE (diffusion, diffusion/advection, burger equation...).

The Model class is pickable, means that it can be sent across the network and between cpu for multiprocessing purpose. It can be save on disk as a binary and reload later. It is important in order to reduce the large compilation overhead. (see Model.save and load_model).

Fields containers
------------------

A special container has been designed to handle initial values of the dependant solutions (the unknowns), the independant variables (spatial coordinates), the constant fields and the post-processed variable (known as helper function).

A factory is linked to the model and is accessible via the model.fields_template property :

.. code-block:: python

    import numpy as np
    from triflow import Model

    model = Model("k * dxxU - c * dxU",
                  "U", ["k", "c"])

    x, dx = np.linspace(0, 1, 100, retstep=True)
    U = np.cos(2 * np.pi * x * 5)
    fields = model.fields_template(x=x, U=U)

The variable involved in the computation are stored on a large vector containing all the fields, and this object give access to each fields to simplify their modification and the computations.

.. code-block:: python

    fields.U[:] = 5
    print(fields.U)
    >>> [5, 5, 5, ..., 5, 5]

Be aware of difference between the attribute giving access to a view of the main array and the one returning a copy of the subarray: the first one allow an on-the-fly modification of the fields (in order to inject boundary condition for exemple), the second one should be only used as read-only meaning.

Numerical scheme, temporal solver
----------------------------------

In order to provide fast and scalable temporal solver, the Jacobian use the `scipy sparse column matrix format`_ (which will reduce the memory usage, especialy for a large number of spatial nodes), and make available the SuperLU_ decomposition, a fast LU sparse matrix decomposition algorithm.

Different temporal schemes are provided in the plugins module:

* a forward Euler scheme
* a backward Euler scheme
* a :math:`\theta` mixed scheme
* A ROW schemes from order 3 up to 6 with fixed and variable time stepping.
* A proxy schemes giving access to all the scipy.integrate.ode schemes.

Each of them have advantages and disadvantages.

They can accept somme extra arguments during their instantiation (for exemple the :math:`\theta` parameter for the :math:`\theta` mixed scheme), and are called with the actual fields, time, time-step, parameters, and accept an optionnal hook modifying fields and parameters each time the solver compute the function or its jacobian.

The following code compute juste one time-step with a Crank-Nicolson scheme.

.. code-block:: python

    import numpy as np
    from triflow import Model, schemes

    model = Model("k * dxxU - c * dxU",
                  "U", ["k", "c"])

    x, dx = np.linspace(0, 1, 100, retstep=True)
    U = np.cos(2 * np.pi * x * 5)
    fields = model.fields_template(x=x, U=U)

    parameters = dict(c=1, k=1, dx=dx)

    t = 0
    dt = 1

    scheme = schemes.Theta(model, theta=.5) # Crank-Nicolson scheme

    new_fields, new_t = scheme(fields, t, dt, parameters)

We obtain with the following code a full resolution up to the target time.

.. code-block:: python

    import numpy as np
    from triflow import Model, schemes

    model = Model("k * dxxU - c * dxU",
                  "U", ["k", "c"])

    x, dx = np.linspace(0, 1, 100, retstep=True)
    U = np.cos(2 * np.pi * x * 5)
    fields = model.fields_template(x=x, U=U)

    parameters = dict(c=1, k=1, dx=dx)

    tmax = 1000
    t = 0
    dt = 1

    scheme = schemes.Theta(model, theta=.5) # Crank-Nicolson scheme

    while t <= tmax:
        fields, t = scheme(fields, t, dt, parameters)

hook and boundary consitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The hook function is used in order to deal with variable and conditional parameters and boundary condition.

Inside the model, the fields are padded in order to solve the equation. If the parameter "periodic" is used, the pad function is used with the mode "wrap" leading to periodic fields. If not, the mode "edge" is used, repeating the first and last node. It is very easy to implement Dirichlet condition with the following function:

.. code-block:: python

    import numpy as np
    from triflow import Model, schemes

    model = Model("k * dxxU",
                  "U", ["k"])

    x, dx = np.linspace(0, 1, 50, retstep=True)
    U = np.cos(2 * np.pi * x * 1.5)
    fields = model.fields_template(x=x, U=U)

    parameters = dict(k=1e-3, dx=dx,
                      time_stepping=True,
                      tol=1E-2, periodic=False)

    tmax = 1
    t = 0
    dt = .01

    scheme = schemes.RODASPR(model)

    def dirichlet_condition(fields, t, pars):
        fields.U[0] = 1
        fields.U[-1] = 1
        return fields, pars

    while t <= tmax:
        fields, t = scheme(fields, t, dt,
                           parameters, hook=dirichlet_condition)

.. _Sympy: http://www.sympy.org/en/index.html
.. _Numpy: http://www.sympy.org/en/index.html
.. _scipy sparse column matrix format: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csc_matrix.html
.. _SuperLU: http://crd-legacy.lbl.gov/~xiaoye/SuperLU/
