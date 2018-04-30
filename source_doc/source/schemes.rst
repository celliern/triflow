Temporal schemes
================

This section will present the structure of a typical temporal scheme and how
to write your own schemes.

List of available schemes
-------------------------

schemes.Theta
^^^^^^^^^^^^^

.. code-block:: python3

    >>> from triflow import schemes
    >>> scheme = schemes.Theta(model, theta)  # doctest:+ELLIPSIS
    ...


This scheme represent a combinaison of the forward and the backward Euler.
With theta = 0, it will be a full forward Euler, with theta = 1, a full
backward Euler and with theta = 0.5, we will have a Crank-Nicolson method.

schemes.scipy_ode
^^^^^^^^^^^^^^^^^

.. code-block:: python3

    >>> from triflow import schemes
    >>> scheme = schemes.scipy_ode(model, integrator, **kwd_integrator)

This scheme is a wrapper around the scipy.integrate.ode.

The integrator is one of these provided by scipy and kwd_integrator allow us
to pass extra parameters to the solver.

Beware that this scheme do not use the sparse jacobian, leading to higher
memory usage and possibly bad performance for large systems. However,
the time-stepping provided is good and is a good choice for validate a
new scheme.

schemes.ROW_general
^^^^^^^^^^^^^^^^^^^
http://www.digibib.tu-bs.de/?docid=00055262
Rang, Joachim: Improved traditional Rosenbrock-Wanner methods for stiff odes
 and daes / Joachim Rang.

This is the parent of all the Rosenbrock-Wanner scheme provided: they follow
the same algorithm with different number of internal steps and different
coefficients. A time-stepping is available and these schemes are suitable for
stiff equations.

* ROS2 (2 steps, only fixed time-step)
* ROS3PRw (3 steps)
* ROS3PRL (4 steps)
* RODASPR (6 steps)


Time stepping
-------------

Time stepping is important to ensure proper computation, especially for stiff
problem. Triflow propose two different implementation, one for the ROW schemes,
and another for the other schemes.

ROW schemes built-in adaptative timestep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROS3PRw, ROS3PRL, RODASPR have a `built-in error computation
<https://en.wikipedia.org/wiki/Runge%E2%80%93
Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods>`_
that allow accurate and automatic time-stepping with neglectible extra cost.
This is not the case for other schemes (except the ones provided by scipy).
Howsoever, an "universal" time stepping algorithm has been implemented
for these schemes.

"Universal" adaptative timestep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the other schemes, error is computed via an extra step : every n steps,
a "coarse" solution for a large step (n * dt) is simulated, and the error is
computed with the difference between the two solutions.

Thus, this extra "coarse" solution can be expensive, especially if the problem
is stiff : for such equation, a scheme with built-in error included should
be prefered.



Internal structure of a scheme
------------------------------

A temporal scheme can be written as any callable object initiated with a model
attribute (which will give access to the system of differential equation to
solve and its jacobian with model.F and model.J).

The `__call__` method have the following signature:

.. code-block:: python3

    t, fields = scheme(t, fields, dt, pars,
                       hook=lambda fields, t, pars: (fields, pars))

It will take as input the actual fields container, the time and the time-step
wanted for this step. As keyword argument it will take a hook, a callable with
the fields, time and parameters as input and fields and parameters as output.
This function give us the ability to make on-the-fly modification of the
fields (for boundary condition), or parameters (allowing time and space
conditional parameters).

This hook has to be called before calling the model routines.

.. code-block:: python3

    class BackwardEuler:

        def __init__(self, model):
            self.model = model

        def __call__(self, t, fields, dt, pars,
                     hook=lambda t, fields, pars: (fields, pars)):
            fields = fields.copy()
            fields, pars = hook(t, fields, pars)
            F = self.model.F(fields, pars)
            J = self.model.J(fields, pars)
            # access the flatten copy of the dependant variables
            U = fields.uflat
            B = dt * (F -  J @ U) + U
            J = (sps.identity(U.size,
                              format='csc') - dt * J)
            # used in order to update the value of the dependant variables
            fields = fields.fill(solver(J, B))
            # We return the hooked fields, be sure that the bdc are taken into account.
            fields, _ = hook(t + dt, fields, pars)
            return t + dt, fields

