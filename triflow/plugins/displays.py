#!/usr/bin/env python
# coding=utf-8

import logging
import warnings
from collections import deque
from uuid import uuid4
from holoviews import Curve, DynamicMap, Layout, streams, Image
from path import Path  # noqa

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())


class TriflowDisplay:
    def __init__(self, skel_data, plot_function):

        self.on_disk = None
        self._plot_pipe = streams.Pipe(data=skel_data)
        self._dynmap = DynamicMap(plot_function, streams=[self._plot_pipe])
        self._writers = []

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
        return self._dynmap + other

    def __mul__(self, other):
        if isinstance(other, TriflowDisplay):
            return self._dynmap * other._dynmap
        self._dynmap * other

    @staticmethod
    def display_fields(simul, keys="all"):
        def plot_function(data):
            nonlocal keys
            curves = []
            keys = data.fields.data_vars if keys == "all" else keys

            for var in keys if not isinstance(keys, str) else [keys]:
                displayed_field = data.fields[var]
                if len(displayed_field.dims) == 1:
                    curves.append(Curve((displayed_field.squeeze()), label=var))
                elif len(displayed_field.dims) == 2:
                    curves.append(
                        Image((displayed_field.squeeze()), label=var))
                else:
                    continue
            return Layout(curves)

        display = TriflowDisplay(simul, plot_function)
        display.connect(simul.stream)
        return display

    @staticmethod
    def display_probe(simul, function,
                      xlabel=None, ylabel=None, buffer=None):
        history = deque([], buffer)
        if not xlabel:
            xlabel = str(uuid4())[:6]
        if not ylabel:
            ylabel = function.__name__
        if ylabel == '<lambda>':
            warnings.warn("Anonymous function used, appending random prefix "
                          "to avoid label confusion")
            ylabel += str(uuid4())[:8]

        def plot_function(data):
            history.append(function(simul))
            return Curve(history, kdims=xlabel, vdims=ylabel)

        display = TriflowDisplay(simul, plot_function)
        display.connect(simul.stream)
        return display
