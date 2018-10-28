#!/usr/bin/env python
# coding=utf-8

import logging
from functools import lru_cache, partial, wraps
from itertools import accumulate, chain
from tempfile import mkdtemp

import attr
import numba
from jinja2 import Template
import numpy as np
from scipy.sparse import csc_matrix

from sympy import (
    And,
    Idx,
    Indexed,
    Integer,
    KroneckerDelta,
    Matrix,
    Number,
    Symbol,
    lambdify,
    oo,
)

from ..grid_builder import GridBuilder
from ..system import PDESys
from .base_compiler import Compiler, register_compiler

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def np_depvar_printer(dvar):
    return Symbol(str(dvar.discrete))


def np_ivar_printer(ivar):
    return (Symbol(str(ivar.discrete)), Symbol(str(ivar.idx)), ivar.step, ivar.N)


def np_Min(args):
    a, b = args
    return np.where(a < b, a, b)


def np_Max(args):
    a, b = args
    return np.where(a < b, b, a)


def np_Heaviside(a):
    return np.where(a < 0, 1, 1)


@register_compiler
@attr.s
class NumbaCompiler(Compiler):
    name = "numba"

    def _convert_inputs(self):
        self.ndim = len(self.system.independent_variables)
        self.dvars = list(map(np_depvar_printer, self.system.dependent_variables))
        self.pars = list(map(np_depvar_printer, self.system.parameters))
        self.ivars, self.idxs, self.steps, self.sizes = zip(
            *map(np_ivar_printer, self.system.independent_variables)
        )
        self.idxs_map = [
            tuple(
                [
                    self.system.independent_variables.index(ivar)
                    for ivar in dvar.independent_variables
                ]
            )
            for dvar in self.system.dependent_variables
        ]

        self.shapes = [
            self.system.shapes[dvar] for dvar in self.system.dependent_variables
        ]

        self.inputs = [
            self.system._t,
            *self.dvars,
            *self.pars,
            *self.ivars,
            *self.steps,
            *self.sizes,
        ]
        self.inputs_cond = [*self.idxs, *self.sizes]

    def _build_evolution_equations(self):
        self._full_exprs = []
        for sys in self.system._system:
            _, exprs = zip(*sys)
            self._full_exprs.extend(exprs)
        template = Template(
            """
@numba.jit(nopython=True, parallel=True, fastmath=False)
def compute_F_vector({% for item in var_names %}{{ item }}, {% endfor %}grid):
    N = grid.shape[0]
    F = np.empty(N)
    for i in numba.prange(N):
        didx, {% for item in var_unpacking %}{{ item }}, {% endfor %}domain, flatidx = grid[i]
        {% for i, expr in exprs %}
        if domain == {{ i }}:
            F[flatidx] = {{ expr }}{% endfor %}
    return F
"""
        )
        var_names = self.inputs
        var_unpacking = self.idxs
        _scope = {}
        exec(template.render(var_names=var_names,
             var_unpacking=var_unpacking,
             exprs=enumerate(self._full_exprs)),
             globals(), _scope)
        self.compute_F_vector = _scope["compute_F_vector"]

        def F(fields, t=0):
            dvars = [
                fields[varname].values
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            pars = [
                fields[varname].values
                for varname in [par.name for par in self.system.parameters]
            ]
            ivars = [
                fields[varname].values
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]
            grid = self.grid_builder.compute_gridinfo(*sizes)
            steps = self.grid_builder.compute_steps(
                tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
            )

            return _scope["compute_F_vector"](t, *dvars, *pars, *ivars, *steps, *sizes, grid)

        self.F = F
    def __attrs_post_init__(self):
        logging.info("fortran compiler: convert_inputs...")
        self._convert_inputs()
        logging.info("fortran compiler: build_evolution_equations...")
        self._build_evolution_equations()
