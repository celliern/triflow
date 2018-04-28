#!/usr/bin/env python
# coding=utf8

import logging
import warnings
from collections import deque
from uuid import uuid4
import multiprocessing as mp

from path import Path
import coolname
from holoviews import Curve, DynamicMap, Layout, streams
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from holoviews.plotting.mpl import MPLRenderer  # noqa

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


INTERACTIVE = is_interactive()
if INTERACTIVE:
    from holoviews import notebook_extension
    notebook_extension("bokeh")


class TriflowDisplay:
    def __init__(self, skel_data, plot_function,
                 on_disk=None, on_disk_folder="./plots/",
                 **renderer_args):

        self.on_disk = None
        self._plot_pipe = streams.Pipe(data=skel_data)
        self._dynmap = (DynamicMap(plot_function,
                                   streams=[self._plot_pipe])
                        .opts("Curve [width=600] {+framewise}"))
        self._writers = []
        if on_disk:
            self._renderer = MPLRenderer.instance()
            target_dir = Path(on_disk_folder)
            target_dir.makedirs_p()

            def save_curves(data):
                target_file = target_dir / "%s_%i" % (on_disk, data.i)
                process = mp.Process(target=self._renderer.save,
                                     args=(self.hv_curve, target_file),
                                     kwargs=renderer_args)
                self._writers.append(process)
                process.start()

            self._plot_pipe.add_subscriber(save_curves)

    def _repr_mimebundle_(self, *args, **kwargs):
        return self.hv_curve._repr_mimebundle_(*args, **kwargs)

    def connect(self, stream):
        stream.sink(self._plot_pipe.send)

    @property
    def hv_curve(self):
        return self._dynmap.collate()

    def __add__(self, other):
        if isinstance(other, TriflowDisplay):
            return self._dynmap + other._dynmap
        self._dynmap + other

    def __mul__(self, other):
        if isinstance(other, TriflowDisplay):
            return self._dynmap * other._dynmap
        self._dynmap * other

    @staticmethod
    def display_fields(simul, keys="all",
                       on_disk=None, on_disk_folder="./plots/",
                       **renderer_args):
        def plot_function(data):
            curves = []
            for var in (data.fields.data_vars if keys == "all" else keys):
                displayed_field = data.fields[var]
                if displayed_field.dims != ('x', ):
                    continue
                curves.append(Curve((displayed_field.squeeze())))
            return Layout(curves).cols(1)
        display = TriflowDisplay(simul, plot_function,
                                 on_disk=on_disk,
                                 on_disk_folder=on_disk_folder,
                                 **renderer_args)
        display.connect(simul.stream)
        return display

    @staticmethod
    def display_probe(simul, function,
                      xlabel=None, ylabel=None, buffer=None,
                      on_disk=None, on_disk_folder="./plots/",
                      **renderer_args):
        history = deque([], buffer)
        if not xlabel:
            xlabel = coolname.generate_slug(2)
        if not ylabel:
            ylabel = function.__name__
        if ylabel == '<lambda>':
            warnings.warn("Anonymous function used, appending random prefix "
                          "to avoid label confusion")
            ylabel += str(uuid4())[:8]

        def plot_function(data):
            history.append(function(simul))
            return Curve(history, kdims=xlabel, vdims=ylabel)
        display = TriflowDisplay(simul, plot_function,
                                 on_disk=on_disk,
                                 on_disk_folder=on_disk_folder,
                                 **renderer_args)
        display.connect(simul.stream)
        return display
