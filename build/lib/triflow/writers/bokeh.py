#!/usr/bin/env python
# coding=utf8

import logging
from itertools import chain

from bokeh.io import output_notebook, push_notebook, show
from bokeh.models.layouts import Column
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure

from triflow.displays import simple_display
from triflow.misc import coroutine


@coroutine
def bokeh_nb_writer(simul):
    output_notebook(hide_banner=True)
    fields = simul.conf.get('bokeh.fields', simul.solver.fields)
    display = simple_display(simul)
    t, field = display.send(simul)
    data = {name: field[name]
            for name
            in simul.solver.fields}
    data['x'] = simul.x
    source = ColumnDataSource(data)

    figs = [figure(title=field, **{key: value
                                   for key, value
                                   in chain(simul.conf.get('bokeh.fig',
                                                           {}).items(),
                                            simul.conf.get('bokeh.fig.%s' %
                                                           field,
                                                           {}).items())})
            for field in fields]
    [fig.line('x', field, source=source,
              **{key: value
                 for key, value
                 in chain(simul.conf.get('bokeh.line',
                                         {}).items(),
                          simul.conf.get('bokeh.line.%s' %
                                         field,
                                         {}).items())})
     for fig, field in zip(figs, fields)]
    handler = show(Column(*figs), notebook_handle=True)
    while True:
        simul = yield
        t, field = display.send(simul)
        data = {name: field[name]
                for name
                in simul.solver.fields}
        data['x'] = simul.x
        source.data = data
        push_notebook(handle=handler)

bokeh_nb_writer.writer_type = 'bokeh'
