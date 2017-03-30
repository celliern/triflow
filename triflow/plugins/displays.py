#!/usr/bin/env python
"""Summary
"""
# coding=utf8

import itertools as it
from collections import deque

import numpy as np
import visdom

def coroutine(func):
    def wrapper(*arg, **kwargs):
        generator = func(*arg, **kwargs)
        next(generator)
        return generator
    return wrapper


@coroutine
def amnesic_mean():
    """Coroutine able to compute mean of a sample without keeping
    data in memory.

    Example
    -------
    In that example we check if the mean value returned by the coroutine feeded
    by normal pseudo random values goes close to 0::
        import numpy as np

        mean_coroutine = amnesic_mean()

        for value in np.random.randn(5000):
            x_mean = mean_coroutine.send(value)
        print(x_mean)
        ... ~0

    """
    increment = yield
    total = increment
    for i in it.count():
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

    Example
    -------
    ::
        from triflow import Model, Simulation
        import numpy as np

        model = Model(funcs="dxxU", vars="U")
        parameters = dict(time_stepping=True, tol=1E-2, dt=1,
                          periodic=True, tmax=10)

        x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
        U = np.cos(x)

        fields = model.fields_template(x=x, U=U)
        simul = Simulation(model, fields, 0, parameters)

        window_save_gen = window_data()

        for fields, t in simul:
            saved_data = window_save_gen.send((fields, t))

    """
    fields, t = yield
    time_list = deque([t], window_len)
    fields_list = deque([fields.rec.T], window_len)
    while True:
        fields, t = yield {'t': np.array(time_list),
                           'fields': np.vstack(fields_list)}
        time_list.append(t)
        fields_list.append(fields.rec.T)


@coroutine
def visdom_update(addr='http://127.0.0.1',
                  env='main',
                  vars=None):
    """Coroutine sending fields data in a interactive plot leaving on a visdom
    server (which has to be launched).

    Parameters
    ----------
    addr: str (default "http://127.0.0.1")
        adress of the visdom server.
    env: str (default "main")
        Visdom environnement where the plot will be send.
    vars: list or None (default None)
        list of the plotted variables. Default all the dependant
        variables of the model.


    Example
    -------
    ::
        from triflow import Model, Simulation
        import numpy as np

        model = Model(funcs="dxxU", vars="U")
        parameters = dict(time_stepping=True, tol=1E-2, dt=1,
                          periodic=True, tmax=10)

        x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
        U = np.cos(x)

        fields = model.fields_template(x=x, U=U)
        simul = Simulation(model, fields, 0, parameters)

        visdom_upt = visdom_update()

        for fields, t in simul:
            visdom_upt.send((fields, t))

    """
    vis = visdom.Visdom(addr)
    fields, t = yield
    wins = {}
    for var in vars if vars else fields.vars:
        wins[var] = vis.line(fields.x,
                             fields[var],
                             env=env,
                             opts=dict(title=var))
    while True:
        fields, t = yield
        for var in vars if vars else fields.vars:
            vis.updateTrace(fields.x, fields[var],
                            wins[var], append=False,
                            env=env,
                            )


@coroutine
def bokeh_fields_update(vars=None, line_kwargs={}, fig_kwargs={}):
    """Coroutine sending fields data in a interactive Bokeh plot displayed in
    a jupyter notebook.
    bokeh.io.output_notebook() have to be called before using this coroutine.

    Parameters
    ----------
    vars: list or None (default None)
        list of the plotted variables. Default all the dependant
        variables of the model.
    line_kwargs: dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments
        passed to the lines plots
    fig_kwargs: dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments
        passed to the figs plots

    Example
    -------
    ::
        from triflow import Model, Simulation
        import numpy as np

        model = Model(funcs="dxxU", vars="U")
        parameters = dict(time_stepping=True, tol=1E-2, dt=1,
                          periodic=True, tmax=10)

        x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
        U = np.cos(x)

        fields = model.fields_template(x=x, U=U)
        simul = Simulation(model, fields, 0, parameters)

        bokeh_upt = bokeh_update()

        for fields, t in simul:
            bokeh_upt.send((fields, t))

    """
    from bokeh.io import push_notebook
    from bokeh.plotting import figure, show, ColumnDataSource
    from bokeh.layouts import Column

    fields, t = yield
    vars = vars if vars else fields.vars
    datasource = ColumnDataSource({var: fields[var]
                                   for var
                                   in list(vars) + ['x']})
    figs = {}
    for var in vars:
        figs[var] = figure(**fig_kwargs.get(var, {}))
        figs[var].line('x', var, source=datasource,
                       **line_kwargs.get(var, {}))
    handler = show(Column(*[figs[var] for var in vars]), notebook_handle=True)
    while True:
        fields, t = yield
        for var in vars:
            datasource.data[var] = fields[var]
        push_notebook(handle=handler)


@coroutine
def bokeh_probes_update(probes, line_kwargs={}, fig_kwargs={}):
    """Coroutine sending custom probes in a interactive Bokeh plot
    displayed in a jupyter notebook.
    bokeh.io.output_notebook() have to be called before using this coroutine.

    Parameters
    ----------
    probes: dictionnary of callable
        Dictionnary with {name: callable} used to plot the probes.
        The signature is the same as in the hooks and return the value
        we want to plot::
            def probe(fields, t, pars):
                ...
                return myprobe
        my probe has to be a single value.

    line_kwargs: dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments
        passed to the lines plots

    fig_kwargs: dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments
        passed to the figs plots

    Example
    -------
    ::
        from triflow import Model, Simulation
        import numpy as np

        model = Model(funcs="dxxU", vars="U")
        parameters = dict(time_stepping=True, tol=1E-2, dt=1,
                          periodic=True, tmax=10)

        x = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
        U = np.cos(x)

        fields = model.fields_template(x=x, U=U)
        simul = Simulation(model, fields, 0, parameters)

        bokeh_upt = bokeh_update()

        for fields, t in simul:
            bokeh_upt.send((fields, t))

    """
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
