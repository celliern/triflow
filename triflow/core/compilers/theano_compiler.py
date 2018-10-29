#!/usr/bin/env python
# coding=utf-8

import logging
from functools import lru_cache, partial, wraps
from itertools import accumulate, chain
from tempfile import mkdtemp

import attr
import numpy as np
import theano.tensor as tt
from joblib import Memory
from scipy.sparse import csc_matrix
from sklearn.tree import DecisionTreeRegressor
from sympy import (
    And,
    Idx,
    Indexed,
    Integer,
    KroneckerDelta,
    Number,
    Symbol,
    oo,
    lambdify,
)
from sympy.printing.theanocode import TheanoPrinter, dim_handling, mapping
from theano import clone, function, scan
from theano.compile.ops import as_op
from theano.ifelse import ifelse
from ..system import PDESys
from ..grid_builder import GridBuilder
from .base_compiler import Compiler, register_compiler

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

cachedir = mkdtemp()
memory = Memory(location=cachedir)


def th_depvar_printer(printer, dvar):
    kwargs = dict(dtypes=printer._dtypes, broadcastables=printer._broadcast)
    return printer.doprint(dvar.discrete, **kwargs)


def th_ivar_printer(printer, ivar):
    kwargs = dict(dtypes=printer._dtypes, broadcastables=printer._broadcast)
    return (
        printer.doprint(ivar.discrete, **kwargs),
        printer.doprint(Symbol(str(ivar.idx)), **kwargs),
        printer.doprint(ivar.step, **kwargs),
        printer.doprint(ivar.N, **kwargs),
    )


def idx_to_symbol(idx, range):
    return Symbol(str(idx))


def theano_and(*args):
    return tt.all(args, axis=0)


mapping[And] = theano_and


class EnhancedTheanoPrinter(TheanoPrinter):
    @wraps(TheanoPrinter.doprint)
    def doprint(self, expr, **kwargs):
        return self._print(expr, dtypes=self._dtypes, broadcastables=self._broadcast)

    def _init_broadcast(self, system):
        dvar_broadcast = {
            dvar.discrete: list(
                dim_handling(
                    [dvar.discrete], dim=len(dvar.independent_variables)
                ).values()
            )
            for dvar in chain(system.dependent_variables, system.parameters)
        }
        dvar_broadcast = {
            key: value[0] if value else tuple() for key, value in dvar_broadcast.items()
        }
        ivar_broadcast = dim_handling(
            list(
                chain(
                    *[
                        (ivar.idx, ivar.discrete)
                        for ivar in system.independent_variables
                    ]
                )
            ),
            dim=1,
        )
        return {
            Symbol(str(key)): value
            for key, value in chain(dvar_broadcast.items(), ivar_broadcast.items())
        }

    def _init_dtypes(self, system):
        sizes_dtypes = {ivar.N: "int32" for ivar in system.independent_variables}
        idx_dtypes = {
            Symbol(str(ivar.idx)): "int32" for ivar in system.independent_variables
        }
        return {
            key: value for key, value in chain(sizes_dtypes.items(), idx_dtypes.items())
        }

    def __init__(self, system, *args, **kwargs):
        self._system = system
        self._broadcast = self._init_broadcast(system)
        self._dtypes = self._init_dtypes(system)
        super().__init__(*args, **kwargs)

    def _print_Idx(self, idx, **kwargs):
        try:
            return tt.constant(
                [self._print_Integer(Integer(str(idx)))], dtype="int32", ndim=1
            )
        except ValueError:
            pass
        idx_symbol = idx.replace(Idx, idx_to_symbol)
        idx_th = self._print(
            idx_symbol, dtypes=self._dtypes, broadcastables=self._broadcast
        )
        return idx_th

    def _print_IndexedBase(self, indexed_base, ndim=1, **kwargs):
        base_symbol = Symbol(str(indexed_base))
        return self._print_Symbol(
            base_symbol, dtypes=self._dtypes, broadcastables=self._broadcast
        )

    def _print_KroneckerDelta(self, kron, **kwargs):
        return ifelse(
            tt.eq(
                *[
                    self._print(
                        arg, dtypes=self._dtypes, broadcastables=self._broadcast
                    )
                    for arg in kron.args
                ]
            ),
            1,
            0,
        )

    def _print_Indexed(self, indexed, **kwargs):
        idxs = [
            self._print(idx, dtypes=self._dtypes, broadcastables=self._broadcast)
            for idx in indexed.indices
        ]
        idxs = [tt.cast(idx, dtype="int32") for idx in idxs]
        th_base = self._print(indexed.base, ndim=len(idxs), **kwargs)
        return th_base[tuple(idxs)]


# def infer_lexsort_shape(node, input_shapes):
#     return [(input_shapes[0][0],)]


# @as_op(itypes=[tt.imatrix], otypes=[tt.ivector], infer_shape=infer_lexsort_shape)
# def th_lexsort(orders):
#     return np.lexsort(orders).astype("int32")


# @as_op(itypes=[tt.ivector], otypes=[tt.imatrix])
# def th_indices(shape):
#     return np.indices(shape, dtype="int32").reshape(len(shape), -1)


@register_compiler
@attr.s
class TheanoCompiler(Compiler):
    name = "theano"
    Printer = EnhancedTheanoPrinter
    system = attr.ib(type=PDESys)
    grid_builder = attr.ib(type=GridBuilder)

    def _convert_inputs(self):
        self.ndim = len(self.system.independent_variables)
        self.dvars = list(
            map(
                partial(th_depvar_printer, self.printer),
                self.system.dependent_variables,
            )
        )
        self.pars = list(
            map(partial(th_depvar_printer, self.printer), self.system.parameters)
        )
        self.ivars, self.idxs, self.th_steps, self.th_sizes = zip(
            *map(
                partial(th_ivar_printer, self.printer),
                self.system.independent_variables,
            )
        )
        self.t = self.printer.doprint(self.system._t)
        self.dvar_idx = tt.bvector("dvar_idx")
        self.idxs_map = [
            tuple(
                [
                    self.system.independent_variables.index(ivar)
                    for ivar in dvar.independent_variables
                ]
            )
            for dvar in self.system.dependent_variables
        ]

        self.inputs = [
            self.t,
            *self.dvars,
            *self.pars,
            *self.ivars,
            self.dvar_idx,
            *self.idxs,
        ]

        self.shapes = [
            list(map(self.printer.doprint, self.system.shapes[dvar]))
            for dvar in self.system.dependent_variables
        ]
        self.shapes = [
            [
                tt.constant(size, dtype="int32") if isinstance(size, int) else size
                for size in shape
            ]
            for shape in self.shapes
        ]

        self.sizes = [
            self.printer.doprint(self.system.sizes[dvar])
            for dvar in self.system.dependent_variables
        ]
        self.size = sum(self.sizes)
        # self.indices = [th_indices(tt.stack(shape)) for shape in self.shapes]
        self.reshaped_idxs = [
            tt.extra_ops.compress(tt.eq(self.dvar_idx, i), tt.stack(self.idxs), axis=1)
            for i in range(self.ndim)
        ]

    def _setup_replacements(self):
        replacement_sizes = {
            size: tt.cast(ivar.size, "int32")
            for ivar, size in zip(self.ivars, self.th_sizes)
        }
        replacement_steps = {
            step: (tt.ptp(ivar) / (ivar.size - 1)).astype("floatX")
            for ivar, step in zip(self.ivars, self.th_steps)
        }
        self._replacement = {**replacement_sizes, **replacement_steps}

    def _build_evolution_equations(self):

        self._full_exprs = []
        for sys in self.system._system:
            _, exprs = zip(*sys)
            self._full_exprs.extend(exprs)

        _subgrids = [tt.imatrix("sub%i" % i) for i in range(len(self._full_exprs))]
        self.evolution_equations = [
            self.printer.doprint(expr) for expr in self._full_exprs
        ]

        F_ = tt.alloc(
            tt.constant(np.nan, dtype="floatX"), (tt.as_tensor_variable(self.size))
        )
        for grid, eq in zip(_subgrids, self.evolution_equations):
            eq = clone(
                eq, replace={self.idxs[i]: grid[:, i + 1] for i in range(self.ndim)}
            )
            F_ = tt.set_subtensor(F_[grid[:, -1]], eq)
        F_routine = function(
            [*self.inputs, *_subgrids],
            F_,
            allow_input_downcast=True,
            accept_inplace=True,
            on_unused_input="ignore",
            givens=self._replacement,
        )

        def F(fields, t=0):
            dvars = [
                fields[varname]
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            pars = [
                fields[varname]
                for varname in [par.name for par in self.system.parameters]
            ]
            ivars = [
                fields[varname]
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]
            subgrids = self.grid_builder.compute_subgrids(*sizes)
            return F_routine(
                t,
                *dvars,
                *pars,
                *ivars,
                *self.grid_builder.compute_idxs(*sizes),
                *subgrids
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
        self._full_jacs_cols = []
        idxs = [ivar.idx for ivar in self.system.independent_variables]
        sizes = [ivar.N for ivar in self.system.independent_variables]
        inputs_cond = [*idxs, *sizes]
        for expr in self._full_exprs:
            wrts = list(filter(self.filter_dvar_indexed, expr.atoms(Indexed)))
            grids = list(map(self.sort_indexed, wrts))
            self._full_jacs_cols.append(
                [lambdify(inputs_cond, grid, modules="numpy") for grid in grids]
            )

        self._full_jacs = []
        for expr in self._full_exprs:
            wrts = list(filter(self.filter_dvar_indexed, expr.atoms(Indexed)))
            diffs = [
                expr.diff(wrt).replace(KroneckerDelta, self._simplify_kron).n()
                for wrt in wrts
            ]
            self._full_jacs.append([self.printer.doprint(diff) for diff in diffs])

        data = []

        _subgrids = [tt.imatrix("sub%i" % i) for i in range(len(self._full_exprs))]

        for grid, jacs in zip(_subgrids, self._full_jacs):
            for jac in jacs:
                J_ = tt.zeros((grid.shape[0],))

                jac = tt.as_tensor_variable(jac)
                jac = clone(
                    jac,
                    replace={self.idxs[i]: grid[:, i + 1] for i in range(self.ndim)},
                )

                jac = tt.set_subtensor(J_[:], jac)
                data.append(jac.reshape((-1,)))

        data = tt.concatenate(data)

        Jdata_routine = function(
            [*self.inputs, *_subgrids],
            data,
            givens=self._replacement,
            on_unused_input="ignore",
            allow_input_downcast=True,
        )

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

        def compute_jacobian_values(t, dvars, pars, ivars, sizes):
            subgrids = self.grid_builder.compute_subgrids(*sizes)
            data = Jdata_routine(
                t,
                *dvars,
                *pars,
                *ivars,
                *self.grid_builder.compute_idxs(*sizes),
                *subgrids
            )
            return data

        def J(fields, t=0):
            dvars = [
                fields[varname]
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            pars = [
                fields[varname]
                for varname in [par.name for par in self.system.parameters]
            ]
            ivars = [
                fields[varname]
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]

            data = compute_jacobian_values(t, dvars, pars, ivars, sizes)
            _, _, perm_rows, indptr, perm, shape = compute_jacobian_coordinates(*sizes)

            return csc_matrix((data[perm], perm_rows, indptr), shape=shape)

        self.J = J

    def __attrs_post_init__(self):
        logging.info("theano compiler: init printer...")
        self.printer = self.Printer(self.system)
        logging.info("theano compiler: convert_inputs...")
        self._convert_inputs()
        logging.info("theano compiler: setup_replacements...")
        self._setup_replacements()
        logging.info("theano compiler: build_evolution_equations...")
        self._build_evolution_equations()
        logging.info("theano compiler: build_jacobian...")
        self._build_jacobian()
