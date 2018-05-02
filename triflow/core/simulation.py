#!/usr/bin/env python
# coding=utf8

import inspect
import logging
import pprint
import time
import warnings
from collections import namedtuple

import pendulum
import streamz
import tqdm
from coolname import generate_slug
from numpy import isclose

from . import schemes
from ..plugins.container import TriflowContainer

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


tqdm = tqdm.tqdm_notebook if is_interactive() else tqdm.tqdm


class Timer:
    def __init__(self, last, total):
        self.last = last
        self.total = total

    def __repr__(self):
        repr = """last:   {last}
total:  {total}"""
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


PostProcess = namedtuple(
    "PostProcess", ["name", "function", "description"])


class Simulation(object):
    """High level container used to run simulation build on triflow Model.
      This object is an iterable which will yield every time step until the
      parameters 'tmax' is reached if provided.
      By default, the solver use a 6th order ROW solver, an implicit method
      with integrated time-stepping.

      Parameters
      ----------
      model : triflow.Model
          Contain finite difference approximation and routine of the dynamical
          system
      fields : triflow.BaseFields or dict (any mappable)
          triflow container or mappable filled with initial conditions
      parameters : dict
          physical parameters of the simulation
      dt : float
          time stepping for output. if time_stepping is False, the internal
          time stepping will be the same.
      t : float, optional, default 0.
          initial time
      tmax : float, optional, default None
          Control the end of the simulation. If None (the default), the com-
          putation will continue until interrupted by the user (using Ctrl-C
          or a SIGTERM signal).
      id : None, optional
          Name of the simulation. A 2 word slug will be generated if not
          provided.
      hook : callable, optional, default null_hook.
          Any callable taking the actual time, fields and parameters and
          return modified fields and parameters.
          Will be called every internal time step and can be used to include
          time dependent or conditionnal parameters, boundary conditions...
          The default null_hook has no impact on the computation.
      scheme : callable, optional, default triflow.schemes.RODASPR
          An callable object which take the simulation state and return
          the next step.
          Its signature is scheme.__call__(fields, t, dt, pars, hook)
          and it should return the next time and the updated fields.
          It take the model and extra positional and named arguments.
      time_stepping : boolean, default True
          Indicate if the time step is controlled by an algorithm dependant of
          the temporal scheme (see the doc on time stepping for extra info).
      **kwargs
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
        status of the simulation, one of the following one:
        ('created', 'running', 'finished', 'failed')
      t : float
        actual time
      tmax : float or None, default None
        stopping time of the simulation. Not stopping if set to None.

      Properties
      ----------
      post_processes: list of triflow.core.simulation.PostProcess
        contain all the post processing function attached to the simulation.
      container: triflow.TriflowContainer
        give access to the attached container, if any.
      timer: triflow.core.simulation.Timer
        return the cpu time of the previous step and the total running time of
        the simulation.
      stream: streamz.Stream
        Streamz starting point, fed by the simulation state after each
        time_step. This interface is used for post-processing, saving the data
        on disk by the TriflowContainer and display the fields in real-time.

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
      >>> simulation = triflow.Simulation(model, fields, pars, dt=5., tmax=50.)
      >>> for t, fields in simulation:
      ...    pass
      >>> print(t)
      50.0
      """  # noqa

    def __init__(self, model, fields, parameters, dt, t=0, tmax=None,
                 id=None, hook=null_hook,
                 scheme=schemes.RODASPR,
                 time_stepping=True, **kwargs):

        def intersection_kwargs(kwargs, function):
            """Inspect the function signature to identify the relevant keys
            in a dictionary of named parameters.
            """
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
        self.user_dt = self.dt = dt
        self.tmax = tmax
        self.i = 0
        self._stream = streamz.Stream()
        self._pprocesses = []

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
        self._iterator = self.compute()

    def _compute_one_step(self, t, fields, pars):
        """
        Compute one step of the simulation, then update the timers.
        """
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
        self.stream.emit(self)

        try:
            while True:
                t, fields, pars = self._compute_one_step(t, fields, pars)

                self.i += 1
                self.t = t
                self.fields = fields
                self.parameters = pars
                for pprocess in self.post_processes:
                    pprocess.function(self)
                self.stream.emit(self)
                yield self.t, self.fields

                if self.tmax and (isclose(self.t, self.tmax)):
                    self._end_simulation()
                    return

        except RuntimeError:
            self.status = 'failed'
            raise

    def _end_simulation(self):
        if self.container:
            self.container.flush()
            self.container.merge()

    def run(self, progress=True, verbose=False):
        """Compute all steps of the simulation. Be careful: if tmax is not set,
        this function will result in an infinit loop.

        Returns
        -------

        (t, fields):
            last time and result fields.
        """
        total_iter = int((self.tmax // self.user_dt) if self.tmax else None)
        log = logging.info if verbose else logging.debug
        if progress:
            with tqdm(initial=(self.i if self.i < total_iter else total_iter),
                      total=total_iter) as pbar:
                for t, fields in self:
                    pbar.update(1)
                    log("%s running: t: %g" % (self.id, t))
                try:
                    return t, fields
                except UnboundLocalError:
                    warnings.warn("Simulation already ended")
        for t, fields in self:
            log("%s running: t: %g" % (self.id, t))
        try:
            return t, fields
        except UnboundLocalError:
            warnings.warn("Simulation already ended")

    def __repr__(self):
        repr = """{simulation_name:=^30}

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
{model_repr}"""
        repr = repr.format(simulation_name=" %s " % self.id,
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

    def attach_container(self, path=None, save="all",
                         mode="w", nbuffer=50, force=False):
        """add a Container to the simulation which allows some
        persistance to the simulation.

        Parameters
        ----------
        path : str or None (default: None)
            path for the container. If None (the default), the data lives only
            in memory (and are available with `simulation.container`)
        mode : str, optional
            "a" or "w" (default "w")
        save : str, optional
            "all" will save every time-step,
            "last" will only get the last time step
        nbuffer : int, optional
            wait until nbuffer data in the Queue before save on disk.
        timeout : int, optional
            wait until timeout since last flush before save on disk.
        force : bool, optional (default False)
            if True, remove the target folder if not empty. if False, raise an
            error.
        """
        self._container = TriflowContainer("%s/%s" % (path, self.id)
                                           if path else None,
                                           save=save,
                                           mode=mode, metadata=self.parameters,
                                           force=force, nbuffer=nbuffer)
        self._container.connect(self.stream)
        return self._container

    @property
    def post_processes(self):
        return self._pprocesses

    @property
    def stream(self):
        return self._stream

    @property
    def container(self):
        return self._container

    @property
    def timer(self):
        return Timer(self._last_running, self._total_running)

    def add_post_process(self, name, post_process, description=""):
        """add a post-process

        Parameters
        ----------
        name : str
            name of the post-traitment
        post_process : callback (function of a class with a __call__ method
                                 or a streamz.Stream).
            this callback have to accept the simulation state as parameter
            and return the modifield simulation state.
            if a streamz.Stream is provided, it will me plugged_in with the
            previous streamz (and ultimately to the initial_stream). All these
            stream accept and return the simulation state.
        description : str, optional, Default is "".
            give extra information about the post-processing
        """

        self._pprocesses.append(PostProcess(name=name,
                                            function=post_process,
                                            description=description))
        self._pprocesses[-1].function(self)

    def remove_post_process(self, name):
        """remove a post-process

        Parameters
        ----------
        name : str
            name of the post-process to remove.
        """
        self._pprocesses = [post_process
                            for post_process in self._pprocesses
                            if post_process.name != name]

    def __iter__(self):
        return self.compute()

    def __next__(self):
        return next(self._iterator)
