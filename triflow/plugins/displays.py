#!/usr/bin/env python
# coding=utf8
"""This module regroups different displays: function and coroutine written
in order to give extra informationto the user during the simulation
(plot, post-processing...)
"""


class bokeh_fields_update():
    """Display fields data in a interactive Bokeh plot displayed in
    a jupyter notebook.

    Parameters
    ----------
    keys : None, optional
        list of the dependant variables to be displayed
    line_kwargs : dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments passed to the lines plots
    fig_kwargs : dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments passed to the figs plots
    init_notebook: True, optional
        if True, initialize the javascript component needed for bokeh.
    """  # noqa

    def __init__(self, simul, keys=None,
                 line_kwargs={},
                 fig_kwargs={},
                 notebook=True):
        from bokeh.io import push_notebook, output_notebook
        from bokeh.plotting import figure, show, ColumnDataSource
        from bokeh.layouts import Column

        if notebook:
            output_notebook()

        setattr(self, '_push', push_notebook)

        keys = keys if keys else [
            key for key in simul.fields._keys if key != 'x']
        self._datasource = ColumnDataSource({key: simul.fields[key]
                                             for key
                                             in list(keys) + ['x']})
        figs = {}
        for key in keys:
            figs[key] = figure(**fig_kwargs.get(key, {}), title=key)
            figs[key].line('x', key, source=self._datasource,
                           **line_kwargs.get(key, {}))
        self._handler = show(Column(*[figs[key]
                                      for key in keys]), notebook_handle=True)
        self._keys = keys

    def __call__(self, t, fields):
        for key in self._keys:
            self._datasource.data[key] = fields[key]
        self._push(handle=self._handler)


class bokeh_probes_update():
    """Display custom probes in a interactive Bokeh plot displayed in a jupyter notebook.

    Parameters
    ----------
    probes : dictionnary of callable
        Dictionnary with {name: callable} used to plot the probes. The signature is the same as in the hooks and return the value we want to plot.
    line_kwargs : dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments passed to the lines plots
    fig_kwargs : dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments passed to the figs plots
    init_notebook: True, optional
        if True, initialize the javascript component needed for bokeh.
    """  # noqa

    def __init__(self, simul, probes,
                 line_kwargs={}, fig_kwargs={},
                 notebook=True):

        from bokeh.io import push_notebook, output_notebook
        from bokeh.plotting import figure, show, ColumnDataSource
        from bokeh.layouts import Column

        if notebook:
            output_notebook()

        setattr(self, '_push', push_notebook)
        self._datasource = ColumnDataSource(dict(t=[simul.t],
                                                 **{name: [probe(simul.t,
                                                                 simul.fields)]
                                                    for name, probe
                                                    in probes.items()}))
        figs = {}
        for name, probe in probes.items():
            figs[name] = figure(**fig_kwargs.get(name, {}), title=name)
            figs[name].line('t', name, source=self._datasource,
                            **line_kwargs.get(name, {}))
        self._handler = show(Column(*[figs[name] for name in probes]),
                             notebook_handle=True)
        self._probes = probes

    def __call__(self, t, fields):
        self._datasource.data['t'].append(t)
        for name, probe in self._probes.items():
            self._datasource.data[name].append(probe(t, fields))
        self._push(handle=self._handler)
