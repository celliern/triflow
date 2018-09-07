#!/usr/bin/env python
# coding=utf-8

import inspect
import logging
import time
from datetime import datetime, timedelta
import warnings
from collections import namedtuple
from uuid import uuid4

import cloudpickle
import streamz
from ..utils import tqdm
from numpy import isclose

from . import schemes
from ..plugins.container import TriflowContainer

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

now = datetime.now


class Timer:
    def __init__(self, last, total):
        self.last = last
        self.total = total

    def __repr__(self):
        repr = """last:   {last}
total:  {total}"""
        return repr.format(
            last=(now() - timedelta(seconds=self.last)),
            total=(now() - timedelta(seconds=self.total)),
        )


def null_hook(t, fields):
    return fields


def get_total_iter(tmax, user_dt):
    if tmax:
        return tmax // user_dt
    return None


def get_initial(total_iter, i):
    if total_iter and i < total_iter:
        return i
    if total_iter:
        return total_iter
    return 0


PostProcess = namedtuple("PostProcess", ["name", "function", "description"])


def _reduce_simulation(
    model,
    fields,
    dt,
    t,
    tmax,
    i,
    id,
    _pprocesses,
    _scheme,
    status,
    _total_running,
    _last_running,
    _created_timestamp,
    _started_timestamp,
    _last_timestamp,
    _actual_timestamp,
    _hook,
    _container,
):
    simul = Simulation(model=model, fields=fields, dt=dt, t=t, tmax=tmax, id=id)
    simul.i = i
    simul._pprocesses = cloudpickle.loads(_pprocesses)
    simul._scheme = cloudpickle.loads(_scheme)
    simul.status = status
    simul._total_running = _total_running
    simul._last_running = _last_running
    simul._created_timestamp = _created_timestamp
    simul._started_timestamp = _started_timestamp
    simul._last_timestamp = _last_timestamp
    simul._actual_timestamp = _actual_timestamp
    simul._hook = _hook
    simul._container = _container

    return simul


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
          Its signature is scheme.__call__(fields, t, dt, hook)
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
      >>> simulation = triflow.Simulation(model, fields, dt=5., tmax=50.)
      >>> for t, fields in simulation:
      ...    pass
      >>> print(t)
      50.0
      """  # noqa

    def __init__(
        self,
        model,
        fields,
        dt,
        t=0,
        tmax=None,
        id=None,
        hook=null_hook,
        scheme=schemes.RODASPR,
        time_stepping=True,
        **kwargs
    ):
        def intersection_kwargs(kwargs, function):
            """Inspect the function signature to identify the relevant keys
            in a dictionary of named parameters.
            """
            func_signature = inspect.signature(function)
            func_parameters = func_signature.parameters
            kwargs = {
                key: value for key, value in kwargs.items() if key in func_parameters
            }
            return kwargs

        kwargs["time_stepping"] = time_stepping
        self.id = str(uuid4())[:6] if not id else id
        self.model = model
        self.fields = model.fields_template(
            **{var: fields[var] for var in fields.variables}
        )
        self.t = t
        self.user_dt = self.dt = dt
        self.tmax = tmax
        self.i = 0
        self._stream = streamz.Stream()
        self._pprocesses = []

        self._scheme = scheme(model, **intersection_kwargs(kwargs, scheme.__init__))
        if time_stepping and not isinstance(
            self._scheme,
            (schemes.RODASPR, schemes.ROS3PRL, schemes.ROS3PRw, schemes.scipy_ode),
        ):
            self._scheme = schemes.time_stepping(
                self._scheme, **intersection_kwargs(kwargs, schemes.time_stepping)
            )
        self.status = "created"

        self._total_running = 0
        self._last_running = 0
        self._created_timestamp = now()
        self._started_timestamp = None
        self._last_timestamp = None
        self._actual_timestamp = now()
        self._hook = hook
        self._container = None
        self._iterator = self.compute()

    def _compute_one_step(self, t, fields):
        """
        Compute one step of the simulation, then update the timers.
        """
        fields = self._hook(t, fields)
        self.dt = self.tmax - t if self.tmax and (t + self.dt >= self.tmax) else self.dt
        before_compute = time.clock()
        t, fields = self._scheme(t, fields, self.dt, hook=self._hook)
        after_compute = time.clock()
        self._last_running = after_compute - before_compute
        self._total_running += self._last_running
        self._last_timestamp = self._actual_timestamp
        self._actual_timestamp = now()
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
        self._started_timestamp = now()
        self.stream.emit(self)

        try:
            while True:
                t, fields = self._compute_one_step(t, fields)

                self.i += 1
                self.t = t
                self.fields = fields
                for pprocess in self.post_processes:
                    pprocess.function(self)
                self.stream.emit(self)
                yield self.t, self.fields

                if self.tmax and isclose(self.t, self.tmax):
                    self._end_simulation()
                    return

        except RuntimeError:
            self.status = "failed"
            raise

    def _end_simulation(self):
        if self.container:
            self.container.flush()
            self.container.merge()

    def _run_with_progress(self, total_iter, initial, log):
        with tqdm(initial=initial, total=total_iter) as pbar:
            for t, fields in self:
                pbar.update(1)
                log("%s running: t: %g" % (self.id, t))
            try:
                return t, fields
            except UnboundLocalError:
                warnings.warn("Simulation already ended")

    def _run_without_progress(self, log):
        for t, fields in self:
            log("%s running: t: %g" % (self.id, t))
        try:
            return t, fields
        except UnboundLocalError:
            warnings.warn("Simulation already ended")

    def run(self, progress=True, verbose=False):
        """Compute all steps of the simulation. Be careful: if tmax is not set,
        this function will result in an infinit loop.

        Returns
        -------

        (t, fields):
            last time and result fields.
        """

        log = logging.info if verbose else logging.debug

        total_iter = get_total_iter(self.tmax, self.user_dt)
        initial = get_initial(total_iter, self.i)

        if progress:
            return self._run_with_progress(total_iter, initial, log)
        return self._run_without_progress(log)

    def __repr__(self):
        repr = """{simulation_name:=^30}

created:      {created_date}
started:      {started_date}
last:         {last_date}

time:         {t:g}
iteration:    {iter:g}

last step:    {step_time}
total time:   {running_time}

Hook function
-------------
{hook_source}

Container
---------
{container}

=========== Model ===========
{model_repr}"""
        repr = repr.format(
            simulation_name=" %s " % self.id,
            container=self.container,
            t=self.t,
            iter=self.i,
            model_repr=self.model,
            hook_source=inspect.getsource(self._hook),
            step_time=self._last_running,
            running_time=self._total_running,
            created_date=(self._created_timestamp),
            started_date=(
                self._started_timestamp if self._started_timestamp else "None"
            ),
            last_date=(self._last_timestamp if self._last_timestamp else "None"),
        )
        return repr

    def attach_container(
        self, path=None, save="all", mode="w", nbuffer=50, force=False
    ):
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
        self._container = TriflowContainer(
            "%s/%s" % (path, self.id) if path else None,
            save=save,
            mode=mode,
            force=force,
            nbuffer=nbuffer,
        )
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

        self._pprocesses.append(
            PostProcess(name=name, function=post_process, description=description)
        )
        self._pprocesses[-1].function(self)

    def remove_post_process(self, name):
        """remove a post-process

        Parameters
        ----------
        name : str
            name of the post-process to remove.
        """
        self._pprocesses = [
            post_process
            for post_process in self._pprocesses
            if post_process.name != name
        ]

    def __iter__(self):
        return self.compute()

    def __next__(self):
        return next(self._iterator)

    def __reduce__(self):
        return (
            _reduce_simulation,
            (
                self.model,
                self.fields,
                self.dt,
                self.t,
                self.tmax,
                self.i,
                self.id,
                cloudpickle.dumps(self._pprocesses),
                cloudpickle.dumps(self._scheme),
                self.status,
                self._total_running,
                self._last_running,
                self._created_timestamp,
                self._started_timestamp,
                self._last_timestamp,
                self._actual_timestamp,
                self._hook,
                self._container,
            ),
        )

