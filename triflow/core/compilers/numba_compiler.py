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
from .numpy_compiler import np_Max, np_Min, np_Heaviside


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def np_depvar_printer(dvar):
    return Symbol(str(dvar.discrete))


def np_ivar_printer(ivar):
    return (Symbol(str(ivar.discrete)), Symbol(str(ivar.idx)), ivar.step, ivar.N)


@numba.jit(nopython=True, parallel=True, fastmath=False)
def Heaviside(a):
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
@numba.jit(nopython=True, parallel=True, fastmath=False, cache=True)
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
        F_routine_str = template.render(
            var_names=var_names,
            var_unpacking=var_unpacking,
            exprs=enumerate(self._full_exprs),
        )
        exec(F_routine_str, globals(), self.__dict__)
        # self.compute_F_vector = _scope["compute_F_vector"]

        def F(fields, t=0):
            dvars = [
                fields[varname].values
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            pars = [
                fields[varname].values
                for varname in [par.name for par in self.system.parameters]
            ]
            pars = [par if par.shape else float(par) for par in pars]
            ivars = [
                fields[varname].values
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]
            grid = self.grid_builder.compute_gridinfo(*sizes)
            steps = self.grid_builder.compute_steps(
                tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
            )

            return self.compute_F_vector(
                t, *dvars, *pars, *ivars, *steps, *sizes, grid
            )

        self.F = F

    def sort_indexed(self, indexed):
        dvar_idx = [dvar.name for dvar in self.system.dependent_variables].index(
            str(indexed.args[0])
        )
        ivars = self.system.dependent_variables[dvar_idx].independent_variables
        idxs = indexed.args[1:]
        idxs = [
            idxs[ivars.index(ivar)] if ivar in ivars else 0
            for ivar in self.system.independent_variables
        ]
        return [dvar_idx, *idxs]

    def filter_dvar_indexed(self, indexed):
        return indexed.base in [
            dvar.discrete for dvar in self.system.dependent_variables
        ]

    def _simplify_kron(self, *kron_args):
        kron = KroneckerDelta(*kron_args)
        return kron.subs({ivar.N: oo for ivar in self.system.independent_variables})

    def _build_jacobian(self):
        self._full_jacs = []
        self._full_jacs_cols = []
        for expr in self._full_exprs:
            wrts = list(filter(self.filter_dvar_indexed, expr.atoms(Indexed)))
            grids = list(map(self.sort_indexed, wrts))
            self._full_jacs_cols.append(
                [
                    lambdify(
                        self.inputs_cond,
                        grid,
                        modules=[
                            {"amax": np_Max, "amin": np_Min, "Heaviside": np_Heaviside},
                            "numpy",
                        ],
                    )
                    for grid in grids
                ]
            )
            diffs = [
                expr.diff(wrt).replace(KroneckerDelta, self._simplify_kron).n()
                for wrt in wrts
            ]
            self._full_jacs.append(diffs)

        template = Template(
                    """
@numba.jit(nopython=True, parallel=True, fastmath=True)
def compute_jacobian_values({% for item in var_names %}{{ item }}, {% endfor %}subgrids, data_size):
    N = len(subgrids)
    data = np.zeros(data_size)
    cursor = 0
    jacs_lenght = {{ jacs_len }}
    
    cursors = [0]
    for i in range(len(subgrids)):
        grid = subgrids[i]
        jl = jacs_lenght[i]
        cursors.append(grid.shape[0] * jl + cursors[-1])
    for i in numba.prange(N):
        cursor = cursors[i]
        grid = subgrids[i]
        M = grid.shape[0]
        {% for jacs in full_jacs %}
        {{ "if" if loop.index==1 else "elif" }} i == {{loop.index0}}:
        {% for jac_func in jacs %}
            next_cursor = cursor + M
            for j in numba.prange(M): {% for item in var_unpacking %}
                {{ item }} = grid[j, {{ loop.index }}]{% endfor %}
                data[cursor + j] = {{ jac_func }}
            cursor = next_cursor
            {% endfor %} {% endfor %}
            
    return data
        """)
            
        var_names = self.inputs
        var_unpacking = self.idxs
        J_routine_str = template.render(
            var_names=var_names,
            var_unpacking=var_unpacking,
            full_jacs=self._full_jacs,
            jacs_len=[len(jacs) for jacs in self._full_jacs],
        )
        exec(J_routine_str, globals(), self.__dict__)

        @lru_cache(maxsize=128)
        def compute_jacobian_coordinates(*sizes):
            subgrids = self.grid_builder.compute_subgrids(*sizes)
            system_sizes = self.grid_builder.compute_sizes(*sizes)
            system_size = sum(system_sizes)

            rows_list = []
            cols_list = []
            for grid, jac_cols in zip(subgrids, self._full_jacs_cols):
                for col_func in jac_cols:
                    cols_ = np.zeros((grid.shape[0], self.ndim + 1), dtype="int32")
                    cols = col_func(*grid[:, 1:-2].T, *sizes)
                    cols = np.stack(
                        [np.broadcast_to(col, cols_.shape[:-1]) for col in cols]
                    )

                    flat_cols = self.grid_builder.get_flat_from_idxs(cols.T, sizes)

                    rows_list.extend(grid[:, -1].reshape((-1,)))
                    cols_list.extend(flat_cols.reshape((-1,)))
            rows = np.array(rows_list)
            cols = np.array(cols_list)

            perm = np.argsort(cols)
            perm_rows = rows[perm]
            perm_cols = cols[perm]
            count = np.zeros((system_size + 1), dtype="int32")
            uq, cnt = np.unique(perm_cols, False, False, True)
            count[uq + 1] = cnt
            indptr = np.cumsum(count)
            return rows, cols, perm_rows, indptr, perm, (system_size, system_size)

        self.compute_jacobian_coordinates = compute_jacobian_coordinates

        def J(fields, t=0):
            dvars = [
                fields[varname].values
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            pars = [
                fields[varname].values
                for varname in [par.name for par in self.system.parameters]
            ]
            pars = [par if par.shape else float(par) for par in pars]
            ivars = [
                fields[varname].values
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]
            steps = self.grid_builder.compute_steps(
                        tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
                    )
            subgrids = self.grid_builder.compute_subgrids(*sizes)
            data_size = sum(
                        [
                            subgrid.shape[0] * len(jacs)
                            for subgrid, jacs in zip(subgrids, self._full_jacs_cols)
                        ]
                    )
            data = self.compute_jacobian_values(t, *dvars, *pars, *ivars, *steps, *sizes, subgrids, data_size)
            _, _, perm_rows, indptr, perm, shape = self.compute_jacobian_coordinates(*sizes)

            return csc_matrix((data[perm], perm_rows, indptr), shape=shape)

        self.J = J

    def __attrs_post_init__(self):
        logging.info("numba compiler: convert_inputs...")
        self._convert_inputs()
        logging.info("numba compiler: build_evolution_equations...")
        self._build_evolution_equations()
        logging.info("numba compiler: build_jacobian...")
        self._build_jacobian()