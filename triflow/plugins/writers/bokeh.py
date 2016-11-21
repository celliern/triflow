#!/usr/bin/env python
# coding=utf8

import logging
from itertools import chain
from multiprocessing import Queue
from threading import Thread

from bokeh.io import output_notebook, push_notebook, show
from bokeh.models.layouts import Column
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure

from triflow.plugins.displays import simple_display


class bokeh_plotter(Thread):
    def __init__(self, simul, queue, display):
        self.simul = simul
        self.queue = queue
        self.display = display
        super().__init__()

    def run(self):
        output_notebook(hide_banner=True)
        simul = self.simul
        fields = simul.conf.get('bokeh.fields', simul.solver.fields)
        t, field = self.display.send(simul)
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
        while not self.simul.stop.is_set():
            t, field = self.queue.get()
            data = {name: field[name]
                    for name
                    in self.simul.solver.fields}
            data['x'] = self.simul.x
            source.data = data
            push_notebook(handle=handler)


def bokeh_nb_writer(simul):
    queue = Queue()
    display = simple_display(simul)
    plotter = bokeh_plotter(simul, queue, display)
    plotter.start()
    for t, fields in display:
        queue.put((t, fields))
        yield


bokeh_nb_writer.writer_type = 'bokeh'
