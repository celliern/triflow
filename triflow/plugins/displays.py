#!/usr/bin/env python
# coding=utf8
"""This module regroups different displays: function and coroutine written
in order to give extra informationto the user during the simulation
(plot, post-processing...)
"""

from collections import deque
from functools import wraps
from itertools import count

import numpy as np


def coroutine(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        generator = func(*arg, **kwargs)
        next(generator)
        return generator
    return wrapper


@coroutine
def amnesic_mean():
    """Coroutine able to compute mean of a sample without keeping data in memory.

      Examples
      --------
      In that example we check if the mean value returned by the coroutine feeded by normal pseudo random values goes close to 0

      >>> import numpy as np
      >>> from triflow.plugins import displays
      >>> mean_coroutine = displays.amnesic_mean()
      >>> for value in np.random.randn(5000):
      ...     x_mean = mean_coroutine.send(value)
      """  # noqa
    increment = yield
    total = increment
    for i in count():
        increment = yield total / (i + 1)
        total += increment


@coroutine
def window_data(window_len=None):
    """Coroutine able to return a dictionnary filled with concatenated values
      of the time and the fields, with optionnal window lenght. Useful to save
      only a part of the solution.

      Parameters
      ----------
      window_len : int or None (default)
          number of last iterations kept. All data kept if None

      Examples
      --------
      >>> from triflow import Model, Simulation, displays
      >>> import numpy as np

      >>> model = Model("dxxU", "U")
      >>> parameters = dict(periodic=True)

      >>> x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
      >>> U = np.cos(x)

      >>> fields = model.fields_template(x=x, U=U)
      >>> simul = Simulation(model, 0, fields, parameters, dt=.5, tmax=10)

      >>> window_save_gen = displays.window_data()

      >>> for t, fields in simul:
      ...     saved_data = window_save_gen.send((t, fields))

      """  # noqa
    t, fields = yield
    time_list = deque([t], window_len)
    fields_list = deque([fields.structured.T], window_len)
    while True:
        t, fields = yield {'t': np.array(time_list),
                           'fields': np.vstack(fields_list)}
        time_list.append(t)
        fields_list.append(fields.structured.T)


@coroutine
def visdom_update(addr='http://127.0.0.1',
                  env='main',
                  keys=None):
    """Coroutine sending fields data in a interactive plot leaving on a visdom
      server (which has to be launched).

      Parameters
      ----------
      addr : str (default "http://127.0.0.1")
          adress of the visdom server.
      env : str (default "main")
          Visdom environnement where the plot will be send.
      keys : None, optional
          Description

      Examples
      --------
      >>> from triflow import Model, Simulation, displays
      >>> import numpy as np

      >>> model = Model("dxxU", "U")
      >>> parameters = dict(periodic=True)

      >>> x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
      >>> U = np.cos(x)

      >>> fields = model.fields_template(x=x, U=U)
      >>> simul = Simulation(model, 0, fields, parameters, dt=.5, tmax=10)
      >>> visdom_upt = displays.visdom_update()
      >>> for t, fields in simul:
      ...     visdom_upt.send((t, fields))

      Deleted Parameters
      ------------------
      vars : list or None (default None)
          list of the plotted variables. Default all the dependant variables of the model.

      """  # noqa
    import visdom
    vis = visdom.Visdom(addr)
    t, fields = yield
    wins = {}
    for key in keys if keys else fields._keys:
        wins[key] = vis.line(fields.x,
                             fields[key],
                             env=env,
                             opts=dict(title=key))
    while True:
        t, fields = yield
        for key in keys if keys else fields._keys:
            vis.updateTrace(fields.x, fields[key],
                            wins[key], append=False,
                            env=env,
                            )


@coroutine
def bokeh_fields_update(keys=None, line_kwargs={}, fig_kwargs={}):
    """Coroutine sending fields data in a interactive Bokeh plot displayed in
      a jupyter notebook.
      bokeh.io.output_notebook() have to be called before using this coroutine.

      Parameters
      ----------
      keys : None, optional
          Description
      line_kwargs : dict of dict
          dictionnary with vars as key and a dictionnary of keywords arguments passed to the lines plots
      fig_kwargs : dict of dict
          dictionnary with vars as key and a dictionnary of keywords arguments passed to the figs plots

      Examples
      --------
      >>> from triflow import Model, Simulation, displays
      >>> import numpy as np

      >>> model = Model("dxxU", "U")
      >>> parameters = dict(periodic=True)

      >>> x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
      >>> U = np.cos(x)

      >>> fields = model.fields_template(x=x, U=U)
      >>> simul = Simulation(model, 0, fields, parameters, dt=.5, tmax=10)

      >>> bokeh_upt = displays.bokeh_fields_update()

      >>> for t, fields in simul:
      ...     bokeh_upt.send((t, fields))

      Deleted Parameters
      ------------------
      vars : list or None (default None)
          list of the plotted variables. Default all the dependant variables of the model.
      """  # noqa
    from bokeh.io import push_notebook
    from bokeh.plotting import figure, show, ColumnDataSource
    from bokeh.layouts import Column

    t, fields = yield
    keys = keys if keys else fields._keys
    datasource = ColumnDataSource({key: fields[key]
                                   for key
                                   in list(keys) + ['x']})
    figs = {}
    for key in keys:
        figs[key] = figure(**fig_kwargs.get(key, {}))
        figs[key].line('x', key, source=datasource,
                       **line_kwargs.get(key, {}))
    handler = show(Column(*[figs[key] for key in keys]), notebook_handle=True)
    while True:
        t, fields = yield
        for key in keys:
            datasource.data[key] = fields[key]
        push_notebook(handle=handler)


@coroutine
def bokeh_probes_update(probes, line_kwargs={}, fig_kwargs={}):
    """Coroutine sending custom probes in a interactive Bokeh plot displayed in a jupyter notebook.
      bokeh.io.output_notebook() have to be called before using this coroutine.

      Parameters
      ----------
      probes : dictionnary of callable
          Dictionnary with {name: callable} used to plot the probes. The signature is the same as in the hooks and return the value we want to plot.
      line_kwargs : dict of dict
          dictionnary with vars as key and a dictionnary of keywords arguments passed to the lines plots
      fig_kwargs : dict of dict
          dictionnary with vars as key and a dictionnary of keywords arguments passed to the figs plots

      Examples
      --------
      >>> from triflow import Model, Simulation, displays
      >>> import numpy as np

      >>> model = Model("dxxU", "U")
      >>> parameters = dict(periodic=True)

      >>> x = np.linspace(-2 * np.pi, 2 * np.pi, 500, endpoint=False)
      >>> U = np.cos(x) * 5

      >>> fields = model.fields_template(x=x, U=U)
      >>> simul = Simulation(model, 0, fields, parameters, dt=.01, tmax=1)
      >>> def mean_probe(t, fields):
      ...     return np.mean(fields.U)
      >>> bokeh_upt = displays.bokeh_probes_update({"mean": mean_probe})

      >>> for t, fields in simul:
      ...     bokeh_upt.send((t, fields))

      """  # noqa
    from bokeh.io import push_notebook
    from bokeh.plotting import figure, show, ColumnDataSource
    from bokeh.layouts import Column
    args = yield
    datasource = ColumnDataSource(dict(t=[args[0]],
                                       **{name: [probe(*args)]
                                          for name, probe
                                          in probes.items()}))
    figs = {}
    for name, probe in probes.items():
        figs[name] = figure(**fig_kwargs.get(name, {}))
        figs[name].line('t', name, source=datasource,
                        **line_kwargs.get(name, {}))
    handler = show(Column(*[figs[name] for name in probes]),
                   notebook_handle=True)
    while True:
        args = yield
        datasource.data['t'].append(args[0])
        for name, probe in probes.items():
            datasource.data[name].append(probe(*args))
        push_notebook(handle=handler)
