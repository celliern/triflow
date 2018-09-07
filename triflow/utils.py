#!/usr/bin/env python
# coding=utf-8

from tqdm import tqdm, tqdm_notebook
import holoviews as hv
from warnings import warn

tqdm = tqdm


def enable_notebook(enable_tqdm=True, enable_bokeh=True, enable_matplotlib=True):
    global tqdm
    if enable_tqdm:
        try:
            import ipywidgets  # noqa
            tqdm = tqdm_notebook
        except ImportError:
            warn("ipywidgets not installed, skipping tqdm_notebook")
    backends = []
    if enable_bokeh:
        try:
            import bokeh  # noqa
            backends.append("bokeh")
        except ImportError:
            warn("bokeh not installed, skipping this backend")
    if enable_matplotlib:
        try:
            import matplotlib  # noqa
            backends.append("matplotlib")
        except ImportError:
            warn("matplotlib not installed, skipping this backend")
    hv.notebook_extension(*backends)
