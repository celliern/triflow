#!/usr/bin/env python
# coding=utf8

import inspect
import logging
import pprint
from functools import partial

import time
import pendulum
from coolname import generate_slug
from path import Path
from . import schemes
from ..plugins import container
from xarray import Dataset


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
                 time_stepping=True,
                 init_bokeh=True,
                 tmax=None, **kwargs):

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
        self.physical_parameters = physical_parameters
        self.fields = model.fields_template(**fields)
        self.t = t
        self.dt = dt
        self.tmax = tmax
        self.i = 0

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
        self._container = None
        self._save = None
        self._displays = []
        self._handler = []
        self._bokeh_layout = []
        self._probes_info = {}
        self._init_bokeh = init_bokeh
        self._iterator = self.compute()
        self.add_probe("ctime", "t", lambda simul: simul.timer.last)
        self.add_probe("full_ctime", "t", lambda simul: simul.timer.total)
        self._compute_probes()

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
        self.fields = fields
        self.t = t
        self.i += 1
        self.physical_parameters = pars
        self._last_timestamp = self._actual_timestamp
        self._actual_timestamp = pendulum.now()
        self._compute_probes()
        if self._container:
            self._save(t, fields, probes=self.probes)
        if self._displays:
            from bokeh.io import push_notebook
            for display in self._displays:
                display(t, fields, probes=self.probes)
            push_notebook(handle=self._handler)
        return t, fields

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
        if self._displays:
            from bokeh.io import push_notebook, show
            from bokeh.plotting import Column
            self._handler = show(Column(*self._bokeh_layout),
                                 notebook_handle=True)
            for display in self._displays:
                display(t, fields, probes=self.probes)
            push_notebook(handle=self._handler)

        try:
            while True:
                if self.tmax and (self.t >= self.tmax):
                    self._end_simul()
                    return
                self._compute_one_step(t, fields, pars)
                yield self.t, self.fields

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
                           step_time=(None if not self._last_running else
                                      pendulum.now()
                                      .subtract(
                                          seconds=self._last_running)
                                      .diff()),
                           running_time=(pendulum.now()
                                         .subtract(
                               seconds=self._total_running)
                               .diff()),
                           container=(None if not
                                      self._container
                                      else self._container.path),
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
        if self._init_bokeh:
            from bokeh.io import output_notebook
            output_notebook()
            self._init_bokeh = False
        display = display(self,
                          *display_args,
                          **display_kwargs)
        self._displays.append(display)
        self._bokeh_layout += display.figs

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
        self._compute_probes()
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
            probes=self._probes,
            mode=mode,
            nbuffer=nbuffer,
            timeout=timeout)
        logging.info("Persistent container attached (%s)"
                     % container_path.abspath())
        if save_iter == "all":
            self._save = self._container.append
        elif save_iter == "last":
            self._save = partial(self._container.set, force=True)
        self._container.metadata["status"] = "created"

    @property
    def container(self):
        return self._container

    @property
    def probes(self):
        return self._probes

    @property
    def timer(self):
        return Timer(self._last_running, self._total_running)

    def __iter__(self):
        return self.compute()

    def __next__(self):
        return next(self._iterator)

    def add_probe(self, name, dims, probe):
        self._probes_info[name] = (dims, probe)

    def _compute_probes(self):
        self._probes = Dataset(data_vars={name: (dims, [probe(self)])
                                          for name, (dims, probe)
                                          in self._probes_info.items()},
                               coords={"t": [self.t],
                                       **{coord: self.fields[coord]
                                          for coord
                                          in self.model._indep_vars}})

    def timer_probe(self):
        return self._last_running
