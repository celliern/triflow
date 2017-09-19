#!/usr/bin/env python
# coding=utf8
"""This module regroups different displays: function and coroutine written
in order to give extra informationto the user during the simulation
(plot, post-processing...)
"""


class display_1D():
    """Display fields data in a interactive Bokeh plot displayed in
    a jupyter notebook.

    Parameters
    ----------
    keys : None, optional
        list of the dependant variables to be displayed if string provided. If tuple or list,
        it should contain (key, callable), and the callable will have the following signature.

        >>> def callable(t, fields, key):
        >>>     field = fields.U + fields.V
        >>>     return field

        That way it is possible to display post-processed data.

    line_kwargs : dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments passed to the lines plots
    fig_kwargs : dict of dict
        dictionnary with vars as key and a dictionnary of keywords arguments passed to the figs plots
    init_notebook: True, optional
        if True, initialize the javascript component needed for bokeh.
    stack: False, optional
        if True, all the plots are displayed in the same figure. fig kwargs is directly passed to this fig.
    """  # noqa

    def __init__(self, simul, keys=None,
                 line_kwargs={},
                 fig_kwargs={},
                 default_fig_kwargs={"width": 600, "height": 250},
                 default_line_kwargs={},
                 notebook=True,
                 stack=False):
        from bokeh.io import push_notebook
        from bokeh.plotting import figure, ColumnDataSource

        setattr(self, '_push', push_notebook)

        keys = keys if keys else [
            key for key in simul.fields.keys() if key != 'x']

        self._datafunc = {'x': lambda t, fields, key: fields.x}
        for key in keys:
            if isinstance(key, str):
                self._datafunc[key] = lambda t, fields, key: fields[key]
            if isinstance(key, (tuple, list)):
                self._datafunc[key[0]] = key[1]
        self._datasource = ColumnDataSource({key:
                                             func(simul.t,
                                                  simul.fields,
                                                  key).values
                                             for (key, func)
                                             in self._datafunc.items()})
        self.keys = list(self._datafunc.keys())
        self.keys.remove("x")

        if stack:
            fig = figure(**default_fig_kwargs)
            for key in self.keys:
                line_config = default_line_kwargs.copy()
                line_config.update(line_kwargs.get(key, {}))
                fig.line('x', key, source=self._datasource,
                         **line_kwargs.get(key, {}))
            self.figs = [fig]
        else:
            figs = {}
            for key in self.keys:
                fig_config = default_fig_kwargs.copy()
                fig_config.update(fig_kwargs.get(key, {}))
                line_config = default_line_kwargs.copy()
                line_config.update(line_kwargs.get(key, {}))
                figs[key] = figure(**fig_config, title=key)
                figs[key].line('x', key, source=self._datasource,
                               **line_config)

            self.figs = [figs[key]
                         for key in self._datafunc.keys()
                         if key != 'x']

    def __call__(self, t, fields, probes):
        for key, func in self._datafunc.items():
            self._datasource.data[key] = func(t, fields, key).values


class display_0D():
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

    def __init__(self, simul, keys=None,
                 line_kwargs={},
                 scatter_kwargs={},
                 fig_kwargs={},
                 default_fig_kwargs={"width": 600, "height": 250},
                 default_line_kwargs={},
                 default_scatter_kwargs={},
                 notebook=True,
                 stack=False):
        from bokeh.io import push_notebook
        from bokeh.plotting import figure, ColumnDataSource

        setattr(self, '_push', push_notebook)

        keys = keys if keys else [
            key for key in simul.probes.keys() if key != 't']

        self._datafunc = {'t': lambda t, probes, key: [t]}
        for key in keys:
            if isinstance(key, str):
                self._datafunc[key] = \
                    lambda t, probes, key: probes[key].values.tolist()
            if isinstance(key, (tuple, list)):
                self._datafunc[key[0]] = key[1]
        self._datasource = ColumnDataSource({key:
                                             func(simul.t,
                                                  simul.probes,
                                                  key)
                                             for (key, func)
                                             in self._datafunc.items()})
        self.keys = list(self._datafunc.keys())
        self.keys.remove("t")

        if stack:
            fig = figure(**default_fig_kwargs)
            for key in self.keys:
                line_config = default_line_kwargs.copy()
                line_config.update(line_kwargs.get(key, {}))
                fig.line('t', key, source=self._datasource,
                         **line_kwargs.get(key, {}))
            self.figs = [fig]
        else:
            figs = {}
            for key in self.keys:
                fig_config = default_fig_kwargs.copy()
                fig_config.update(fig_kwargs.get(key, {}))
                line_config = default_line_kwargs.copy()
                line_config.update(line_kwargs.get(key, {}))
                figs[key] = figure(**fig_config, title=key)
                figs[key].line('t', key, source=self._datasource,
                               **line_config)
                figs[key].scatter('t', key, source=self._datasource)
            self.figs = [figs[key]
                         for key in self._datafunc.keys()
                         if key != 't']

    def __call__(self, t, fields, probes):
        for key, func in self._datafunc.items():
            self._datasource.data[key].append(func(t, probes, key)[0])
