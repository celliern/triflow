#!/usr/bin/env python
# coding=utf8

import itertools as it
import logging
import inspect

from coolname import generate_slug
from triflow.plugins import schemes

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Simulation(object):
    """High level container used to run simulation build on triflow Model.
      This object is an iterable which will yield every time step until the parameters 'tmax' is reached if provided.
      By default, the solver use a 6th order ROW solver, an implicit method with integrated time-stepping.

      Parameters
      ----------
      model : triflow.Model
          Contain finite difference approximation and routine of the dynamical system
      t : float
          initial time
      fields : triflow.Fields
          triflow container filled with initial conditions
      physical_parameters : dict
          physical parameters of the simulation
      id : None, optional
          name of the simulation. A 2 word slug will be generated if not provided.
      hook : callable, optional
          any callable taking the actual time, fields and parameters and return modified fields and parameters. Will be called every internal time step and can be used to include time dependent or conditionnal parameters, boundary conditions...
      scheme : callable, optional, default triflow.schemes.RODASPR
          an callable object which take the simulation state and return the next step. Its signature is scheme.__call__(fields, t, dt, pars, hook) and it should return the next time and the updated fields. It take the model and extra positional and named arguments.
      *args, **kwargs
          extra arguments passed to the scheme.
      *args, **kwargs
          extra arguments passed to the scheme.

      Attributes
      ----------
      dt : float
        output time step
      fields : triflow.Fields
        triflow container filled with actual data
      i : int
        actual iteration
      id : str
        name of the simulation
      model : triflow.Model
        triflow Model used in the simulation
      physical_parameters : dict
        physical parameters of the simulation
      status : str
        status of the simulation, one of the following one: ('created', 'running', 'finished', 'failed')
      t : float
        actual time
      tmax : float or None, default None
        stopping time of the simulation. Not stopping if set to None.

      Examples
      --------
      >>> import numpy as np
      >>> import triflow
      >>> model = triflow.Model(["k1 * dxxU",
      ...                        "k2 * dxxV"],
      ...                       ["U", "V"],
      ...                       ["k1", "k2"])
      >>> x = np.linspace(0, 100, 1000, endpoint=False)
      >>> U = np.cos(x * 2 * np.pi / 100)
      >>> V = np.sin(x * 2 * np.pi / 100)
      >>> fields = model.fields_template(x=x, U=U, V=V)
      >>> pars = {'k1': 1, 'k2': 1, 'periodic': True}
      >>> simulation = triflow.Simulation(model, 0, fields,
      ...                                 pars, dt=5, tmax=50)
      >>> for t, fields in simulation:
      ...    pass
      >>> print(t)
      50
      """  # noqa

    def __init__(self, model, t, fields, physical_parameters, dt,
                 id=None, hook=lambda t, fields, pars: (fields, pars),
                 scheme=schemes.RODASPR,
                 tmax=None, **kwargs):

        def intersection_kwargs(kwargs, function):
            func_signature = inspect.signature(function)
            func_parameters = func_signature.parameters
            kwargs = {key: value
                      for key, value
                      in kwargs.items() if key in func_parameters}
            return kwargs

        self.id = generate_slug(2) if not id else id
        self.model = model
        self.physical_parameters = physical_parameters

        self.fields = fields
        self.t = t
        self.dt = dt
        self.tmax = tmax
        self.i = 0

        self._scheme = scheme(model, **intersection_kwargs(kwargs,
                                                           scheme.__init__))
        self.status = 'created'
        self._hook = hook
        self._iterator = self.compute()

    def compute(self):
        """Generator which yield the actual state of the system every dt.

        Yields
        ------
        tuple : t, fields
            Actual time and updated fields container.
        """
        fields = self.fields
        t = self.t
        pars = self.physical_parameters
        try:
            while True:
                fields, pars = self._hook(t, fields, pars)
                t, fields = self._scheme(t, fields, self.dt,
                                         pars, hook=self._hook)
                self.fields = fields
                self.t = t
                self.physical_parameters = pars
                yield t, fields
        except RuntimeError:
            self.status = 'failed'
            raise

    def _takewhile(self, outputs):

        if self.tmax is None:
            return True
        if self.t > self.tmax:
            self.status = 'finished'
            return False
        return True

    def __iter__(self):
        return it.takewhile(self._takewhile, self.compute())

    def __next__(self):
        return next(self.iterator)
