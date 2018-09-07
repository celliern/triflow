#!/usr/bin/env python
# coding=utf-8

import logging
from functools import lru_cache, partial, wraps
from itertools import accumulate, chain
from tempfile import mkdtemp

import attr
import numpy as np
import theano.tensor as tt
from scipy.sparse import csc_matrix
from sympy import (
    Matrix,
    And,
    Idx,
    Indexed,
    Integer,
    KroneckerDelta,
    Number,
    Symbol,
    lambdify,
    oo,
)
from ..system import PDESys
from ..grid_builder import GridBuilder

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def np_depvar_printer(dvar):
    return Symbol(str(dvar.discrete))


def np_ivar_printer(ivar):
    return (Symbol(str(ivar.discrete)), Symbol(str(ivar.idx)), ivar.step, ivar.N)


@attr.s
class NumpyCompiler:
    system = attr.ib(type=PDESys)
    grid_builder = attr.ib(type=GridBuilder)

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
            *self.idxs,
            *self.steps,
            *self.sizes,
        ]
        self.inputs_cond = [*self.idxs, *self.sizes]

    def _build_evolution_equations(self):
        self._full_exprs = []
        for sys in self.system._system:
            _, exprs = zip(*sys)
            self._full_exprs.extend(exprs)
        evolution_equations = [
            lambdify(self.inputs, expr.n(), modules="numpy")
            for expr in self._full_exprs
        ]

        def compute_F_vector(t, dvars, pars, ivars, sizes):
            subgrids = self.grid_builder.compute_subgrids(*sizes)
            steps = self.grid_builder.compute_steps(
                tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
            )
            system_sizes = self.grid_builder.compute_sizes(*sizes)
            system_size = sum(system_sizes)
            F = np.empty(system_size)
            for grid, eq in zip(subgrids, evolution_equations):
                Fi = eq(t, *dvars, *pars, *ivars, *grid[:, 1:-2].T, *steps, *sizes)
                F[grid[:, -1]] = Fi
            return F

        self.compute_F_vector = compute_F_vector

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

            return compute_F_vector(t, dvars, pars, ivars, sizes)

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
                [lambdify(self.inputs_cond, grid, modules="numpy") for grid in grids]
            )
            diffs = [
                expr.diff(wrt).replace(KroneckerDelta, self._simplify_kron).n()
                for wrt in wrts
            ]
            self._full_jacs.append(
                [lambdify(self.inputs, diff, modules="numpy") for diff in diffs]
            )

        def compute_jacobian_values(t, dvars, pars, ivars, sizes):
            subgrids = self.grid_builder.compute_subgrids(*sizes)
            steps = self.grid_builder.compute_steps(
                tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
            )
            data_size = sum(
                [
                    subgrid.shape[0] * len(jacs)
                    for subgrid, jacs in zip(subgrids, self._full_jacs_cols)
                ]
            )
            data = np.zeros(data_size)

            cursor = 0
            for grid, jacs in zip(subgrids, self._full_jacs):
                for jac_func in jacs:
                    next_cursor = cursor + grid.shape[0]
                    jac = jac_func(
                        t, *dvars, *pars, *ivars, *grid[:, 1:-2].T, *steps, *sizes
                    )
                    data[cursor:next_cursor] = jac
                    cursor = next_cursor
            return data

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

        self.compute_jacobian_values = compute_jacobian_values
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
            ivars = [
                fields[varname].values
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]

            data = compute_jacobian_values(t, dvars, pars, ivars, sizes)
            _, _, perm_rows, indptr, perm, shape = compute_jacobian_coordinates(*sizes)

            return csc_matrix((data[perm], perm_rows, indptr), shape=shape)

        self.J = J

    def __attrs_post_init__(self):
        logging.info("numpy compiler: convert_inputs...")
        self._convert_inputs()
        logging.info("numpy compiler: build_evolution_equations...")
        self._build_evolution_equations()
        logging.info("numpy compiler: build_jacobian...")
        self._build_jacobian()


# system = trf.core.system.PDESys(
#     ["dxxU + dyyU", "dxU + dyyV"],
#     ["U(x, y)", "V(x, y)"],
#     parameters=[],
#     boundary_conditions=dict(U=dict(x="periodic")),
# )

# N_x = 10
# N_y = 7
# x = np.linspace(-1, 1, N_x, endpoint=False)
# y = np.linspace(-1, 1, N_y, endpoint=False)
# U = np.cos(2 * np.pi * x[:, None]) + y[None, :] * 0
# V = np.cos(2 * np.pi * y[None, :]) + x[:, None] * 0

# x_idx, y_idx = np.indices([N_x, N_y])

# comp = NumpyCompiler(system)
# assert (
#     comp._compute_shapes(x.size, y.size) == np.array([[N_x, N_y], [N_x, N_y]])
# ).all()
# assert (comp._compute_sizes(x.size, y.size) == np.array([N_x * N_y, N_x * N_y])).all()
# block1 = np.block(
#     [
#         [0, *[1] * (N_y - 2), 2],
#         *[[3, *[4] * (N_y - 2), 5]] * (N_x - 2),
#         [6, *[7] * (N_y - 2), 8],
#     ]
# )
# block2 = block1 + 9
# assert (comp._compute_domains(x.size, y.size) == np.stack([block1, block2])).all()
# # assert ... self.compute_gridinfo = compute_gridinfo
# # assert ... self.compute_flat_maps = compute_flat_maps
# # assert ... self.compute_subgrids = compute_subgrids
# # assert ... self.compute_dvars_to_flat = compute_dvars_to_flat
# # assert ... self.compute_flat_to_dvars = compute_flat_to_dvars
