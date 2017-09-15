#!/usr/bin/env python
# coding=utf8

import inspect
import logging
import pprint

import pendulum
from coolname import generate_slug
from path import Path
from triflow.plugins import container, schemes
from triflow.core.fields import BaseFields


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


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
                 id=None, hook=null_hook,
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
        if not isinstance(fields, BaseFields):
            fields = model.fields_template(**fields)
        self.fields = fields
        self.t = t
        self.dt = dt
        self.tmax = tmax
        self.i = 0

        self._scheme = scheme(model, **intersection_kwargs(kwargs,
                                                           scheme.__init__))
        self.status = 'created'

        self._created_timestamp = pendulum.now()
        self._started_timestamp = None
        self._last_timestamp = None
        self._actual_timestamp = pendulum.now()
        self._hook = hook
        self._container = None
        self._displays = []
        self._iterator = self.compute()

    # @staticmethod
    # def restart_simulation(model, path, id):
    #     old_container = container.TreantContainer(Path(path) / id, mode="r")
    #     fields = old_container.data
    #     return Simulation(model, )

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
        self._started_timestamp = pendulum.now()
        for display in self._displays:
            display(t, fields)
        try:
            while True:
                fields, pars = self._hook(t, fields, pars)
                self.dt = (self.tmax - t
                           if self.tmax and (t + self.dt >= self.tmax)
                           else self.dt)
                t, fields = self._scheme(t, fields, self.dt,
                                         pars, hook=self._hook)
                self.fields = fields
                self.t = t
                self.i += 1
                self.physical_parameters = pars
                self._last_timestamp = self._actual_timestamp
                self._actual_timestamp = pendulum.now()
                for display in self._displays:
                    display(self.t, self.fields)
                if self._container:
                    self._container.append(t, fields)
                yield self.t, self.fields
                if self.tmax and (self.t >= self.tmax):
                    self._end_simul()
                    return
        except RuntimeError:
            self.status = 'failed'
            if self._container:
                self._container.metadata["status"] = "failed"
            raise

    def _end_simul(self):
        if self._container:
            self._container.metadata["status"] = "finished"
            self._container.close()

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

container:    {container}

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
                                in self.physical_parameters.items()]),
                           t=self.t,
                           iter=self.i,
                           model_repr=self.model,
                           hook_source=inspect.getsource(self._hook),
                           step_time=(None
                                      if not self._last_timestamp
                                      else self._last_timestamp.diff(
                                          self._actual_timestamp)),
                           running_time=(None
                                         if not self._started_timestamp
                                         else self._started_timestamp.diff(
                                             self._actual_timestamp)),
                           container=(self._container.path),
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

    def add_display(self, display, *display_args, **display_kwargs):
        """add a display for the simulation.

        Parameters
        ----------
        display : callable
            a display as the one available in triflow.displays
        *display_args
            positional arguments for the display function (other than the simulation itself)
        **display_kwargs
            named arguments for the display function
        """  # noqa
        self._displays.append(display(self, *display_args, **display_kwargs))

    def attach_container(self, path="output/", mode="w",
                         nbuffer=50, timeout=180, force=False):
        """add a Container to the simulation which allows some
        persistance to the simulation.

        Parameters
        ----------
        path : str
            path for the container
        mode : str, optional
            "a" or "w"
        nbuffer : int, optional
            wait until nbuffer data in the Queue before save on disk.
        timeout : int, optional
            wait until timeout since last flush before save on disk.
        """
        container_path = Path(path) / self.id
        self._container = container.Container(
            container_path,
            t0=self.t,
            initial_fields=self.fields,
            force=force,
            metadata=dict(id=self.id, dt=self.dt, tmax=self.tmax,
                          timestamp="{:%Y-%m-%d %H:%M:%S}"
                          .format(self._created_timestamp),
                          hook=inspect.getsource(self._hook),
                          **self.physical_parameters),
            mode=mode,
            nbuffer=nbuffer,
            timeout=timeout)
        logging.info("Persistent container attached (%s)"
                     % container_path.abspath())
        self._container.metadata["status"] = "created"

    @property
    def container(self):
        return self._container

    def __iter__(self):
        return self.compute()

    def __next__(self):
        return next(self._iterator)
