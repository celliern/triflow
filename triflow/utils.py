#!/usr/bin/env python
# coding=utf-8

from tqdm import tqdm, tqdm_notebook
import holoviews as hv
from warnings import warn
from itertools import chain
from sympy import Dummy, Indexed, Idx, Expr, solve
from functools import wraps, partial
from boltons.iterutils import remap, default_enter
import logging

tqdm = tqdm

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())


def generate_dummy_map(exprs):
    to_dummify = set(chain(*[expr.atoms(Indexed, Idx) for expr in exprs]))
    dummy_map = {var: Dummy() for var in to_dummify}
    reverse_dummy_map = {dummy: var for var, dummy in dummy_map.items()}
    return dummy_map, reverse_dummy_map


@wraps(solve)
def solve_with_dummy(exprs, var, *args, **kwargs):
    def visit_dummy(subs, path, key, value):
        try:
            value = value.subs(subs)
        except AttributeError:
            pass
        try:
            key = key.subs(subs)
        except AttributeError:
            pass
        return key, value

    dummy_map, reverse_dummy_map = generate_dummy_map(exprs)
    dummy_exprs = [expr.subs(dummy_map) for expr in exprs]
    dummy_var = var.subs(dummy_map)
    dummy_sol = solve(dummy_exprs, dummy_var)
    sol = remap(dummy_sol, visit=partial(visit_dummy, reverse_dummy_map))
    return sol


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
