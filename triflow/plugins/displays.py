#!/usr/bin/env python
# coding=utf8

import logging
import warnings
from collections import deque
from uuid import uuid4

import coolname
from holoviews import Curve, DynamicMap, Layout, streams

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())


class TriflowDisplay:
    def __init__(self, simul, plot_function, connect=True):
        self._plot_pipe = streams.Pipe(data=simul)
        self._dynmap = (DynamicMap(plot_function,
                                   streams=[self._plot_pipe])
                        .opts("Curve [width=600] {+framewise}"))
        if connect:
            self.connect(simul.stream)

    def connect(self, stream):
        stream.sink(self._plot_pipe.send)

    def show(self):
        return self._dynmap

    def __add__(self, other):
        if isinstance(other, TriflowDisplay):
            return self._dynmap + other._dynmap
        self._dynmap + other

    def __mul__(self, other):
        if isinstance(other, TriflowDisplay):
            return self._dynmap * other._dynmap
        self._dynmap * other

    @staticmethod
    def display_fields(simul, keys="all"):
        def plot_function(data):
            curves = []
            for var in (data.fields.data_vars if keys == "all" else keys):
                displayed_field = data.fields[var]
                if displayed_field.dims != ('x', ):
                    continue
                curves.append(Curve((displayed_field.squeeze())))
            return Layout(curves).cols(1)
        return TriflowDisplay(simul, plot_function)

    @staticmethod
    def display_probe(simul, function, xlabel=None, ylabel=None, buffer=None):
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
        return TriflowDisplay(simul, plot_function)
