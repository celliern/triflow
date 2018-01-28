#!/usr/bin/env python
# coding=utf8

import inspect
import logging
import pprint

import time
import streamz
import pendulum
from coolname import generate_slug
from . import schemes


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Timer:
    def __init__(self, last, total):
        self.last = last
        self.total = total

    def __repr__(self):
        repr = """
last:   {last}
total:  {total}
"""
        return repr.format(last=(pendulum.now()
                                 .subtract(
            seconds=self.last)
            .diff()),
            total=(pendulum.now()
                   .subtract(
                seconds=self.total)
            .diff()))


def null_hook(t, fields, pars):
    return fields, pars


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
      fields : triflow.BaseFields or dict (any mappable)
          triflow container or mappable filled with initial conditions
      parameters : dict
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
      parameters : dict
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

    def __init__(self, model, fields, parameters, dt, t=0, tmax=None,
                 id=None, hook=null_hook,
                 scheme=schemes.RODASPR,
                 time_stepping=True, **kwargs):

        def intersection_kwargs(kwargs, function):
            func_signature = inspect.signature(function)
            func_parameters = func_signature.parameters
            kwargs = {key: value
                      for key, value
                      in kwargs.items() if key in func_parameters}
            return kwargs
        kwargs["time_stepping"] = time_stepping
        self.id = generate_slug(2) if not id else id
        self.model = model
        self.parameters = parameters
        self.fields = model.fields_template(**fields)
        self.t = t
        self.dt = dt
        self.tmax = tmax
        self.i = 0
        self.stream = streamz.Stream()

        self._scheme = scheme(model,
                              **intersection_kwargs(kwargs,
                                                    scheme.__init__))
        if (time_stepping and
            self._scheme not in [schemes.RODASPR,
                                 schemes.ROS3PRL,
                                 schemes.ROS3PRw]):
            self._scheme = schemes.time_stepping(
                self._scheme,
                **intersection_kwargs(kwargs,
                                      schemes.time_stepping))
        self.status = 'created'

        self._total_running = 0
        self._last_running = 0
        self._created_timestamp = pendulum.now()
        self._started_timestamp = None
        self._last_timestamp = None
        self._actual_timestamp = pendulum.now()
        self._hook = hook
        self._iterator = self.compute()

    def _compute_one_step(self, t, fields, pars):
        fields, pars = self._hook(t, fields, pars)
        self.dt = (self.tmax - t
                   if self.tmax and (t + self.dt >= self.tmax)
                   else self.dt)
        before_compute = time.clock()
        t, fields = self._scheme(t, fields, self.dt,
                                 pars, hook=self._hook)
        after_compute = time.clock()
        self._last_running = after_compute - before_compute
        self._total_running += self._last_running
        self._last_timestamp = self._actual_timestamp
        self._actual_timestamp = pendulum.now()
        return t, fields, pars

    def compute(self):
        """Generator which yield the actual state of the system every dt.

        Yields
        ------
        tuple : t, fields
            Actual time and updated fields container.
        """
        fields = self.fields
        t = self.t
        pars = self.parameters
        self._started_timestamp = pendulum.now()

        try:
            while True:
                if self.tmax and (self.t >= self.tmax):
                    self._end_simul()
                    return
                t, fields, pars = self._compute_one_step(t, fields, pars)

                self.i += 1
                self.t = t
                self.fields = fields
                self.parameters = pars

                self.stream.emit(self)

                yield self.t, self.fields

        except RuntimeError:
            self.status = 'failed'

    def _end_simul(self):
        pass

    def __repr__(self):
        repr = """
{simul_name:=^30}

created:      {created_date}
started:      {started_date}
last:         {last_date}

time:         {t:g}
iteration:    {iter:g}

last step:    {step_time}
total time:   {running_time}


Physical parameters
-------------------
{parameters}

Hook function
-------------
{hook_source}

=========== Model ===========
{model_repr}

"""
        repr = repr.format(simul_name=" %s " % self.id,
                           parameters="\n\t".join(
                               [("%s:" % key).ljust(12) +
                                pprint.pformat(value)
                                for key, value
                                in self.parameters.items()]),
                           t=self.t,
                           iter=self.i,
                           model_repr=self.model,
                           hook_source=inspect.getsource(self._hook),
                           step_time=(None if not self._last_running else
                                      pendulum.now()
                                      .subtract(
                                          seconds=self._last_running)
                                      .diff()),
                           running_time=(pendulum.now()
                                         .subtract(
                               seconds=self._total_running)
                               .diff()),
                           created_date=(self._created_timestamp
                                         .to_cookie_string()),
                           started_date=(self._started_timestamp
                                         .to_cookie_string()
                                         if self._started_timestamp
                                         else "None"),
                           last_date=(self._last_timestamp
                                      .to_cookie_string()
                                      if self._last_timestamp
                                      else "None"))
        return repr

    def attach_container(self, path="output/", save_iter="all",
                         mode="w", nbuffer=50, timeout=180, force=False):
        """add a Container to the simulation which allows some
        persistance to the simulation.

        Parameters
        ----------
        path : str
            path for the container
        mode : str, optional
            "a" or "w" (default "w")
        mode : str, optional
            "all" will save every time-step,
            "last" will only get the last time step
        nbuffer : int, optional
            wait until nbuffer data in the Queue before save on disk.
        timeout : int, optional
            wait until timeout since last flush before save on disk.
        """
        pass

    @property
    def timer(self):
        return Timer(self._last_running, self._total_running)

    def __iter__(self):
        return self.compute()

    def __next__(self):
        return next(self._iterator)
