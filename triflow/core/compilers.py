#!/usr/bin/env python
# coding=utf8

from functools import lru_cache, partial, wraps
from itertools import accumulate, chain
from tempfile import mkdtemp

import attr
import numpy as np
import theano as th
import theano.tensor as tt
from joblib import Memory
from scipy.sparse import csc_matrix
from sklearn.tree import DecisionTreeRegressor
from sympy import (And, Idx, Indexed, Integer, KroneckerDelta, Number, Symbol,
                   oo)
from sympy.printing.theanocode import TheanoPrinter, dim_handling, mapping
from theano import function
from theano.compile.ops import as_op
from triflow.core.system import PDESys

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
        return self._print(
            expr, dtypes=self._dtypes, broadcastables=self._broadcast)

    def _init_broadcast(self, system):
        dvar_broadcast = {
            dvar.discrete: list(
                dim_handling(
                    [dvar.discrete],
                    dim=len(dvar.independent_variables)).values())[0]
            for dvar in system.dependent_variables
        }
        ivar_broadcast = dim_handling(
            list(
                chain(*[(ivar.idx, ivar.discrete)
                        for ivar in system.independent_variables])),
            dim=1,
        )
        return {
            Symbol(str(key)): value
            for key, value in chain(dvar_broadcast.items(),
                                    ivar_broadcast.items())
        }

    def _init_dtypes(self, system):
        sizes_dtypes = {
            ivar.N: "int32"
            for ivar in system.independent_variables
        }
        idx_dtypes = {
            Symbol(str(ivar.idx)): "int32"
            for ivar in system.independent_variables
        }
        return {
            key: value
            for key, value in chain(sizes_dtypes.items(), idx_dtypes.items())
        }

    def __init__(self, system, *args, **kwargs):
        self._system = system
        self._broadcast = self._init_broadcast(system)
        self._dtypes = self._init_dtypes(system)
        super().__init__(*args, **kwargs)

    def _print_Idx(self, idx, **kwargs):
        try:
            return tt.constant(
                [self._print_Integer(Integer(str(idx)))],
                dtype="int32",
                ndim=1)
        except ValueError:
            pass
        idx_symbol = idx.replace(Idx, idx_to_symbol)
        idx_th = self._print(
            idx_symbol, dtypes=self._dtypes, broadcastables=self._broadcast)
        return idx_th

    def _print_IndexedBase(self, indexed_base, ndim=1, **kwargs):
        base_symbol = Symbol(str(indexed_base))
        return self._print_Symbol(
            base_symbol, dtypes=self._dtypes, broadcastables=self._broadcast)

    def _print_KroneckerDelta(self, kron, **kwargs):
        return th.ifelse.ifelse(
            tt.eq(*[
                self._print(
                    arg, dtypes=self._dtypes, broadcastables=self._broadcast)
                for arg in kron.args
            ]),
            1,
            0,
        )

    def _print_Indexed(self, indexed, **kwargs):
        idxs = [
            self._print(
                idx, dtypes=self._dtypes, broadcastables=self._broadcast)
            for idx in indexed.indices
        ]
        idxs = [tt.cast(idx, dtype="int32") for idx in idxs]

        th_base = self._print(indexed.base, ndim=len(idxs), **kwargs)

        return th_base[tuple(idxs)]


def infer_lexsort_shape(node, input_shapes):
    return [(input_shapes[0][0], )]


@as_op(
    itypes=[tt.imatrix], otypes=[tt.ivector], infer_shape=infer_lexsort_shape)
def th_lexsort(orders):
    return np.lexsort(orders)


@as_op(itypes=[tt.ivector], otypes=[tt.imatrix])
def th_indices(shape):
    return np.indices(shape, dtype="int32").reshape(len(shape), -1)


@attr.s
class TheanoCompiler:
    Printer = EnhancedTheanoPrinter
    system = attr.ib(type=PDESys)

    def _convert_inputs(self):
        self.ndim = len(self.system.independent_variables)
        self.dvars = list(
            map(
                partial(th_depvar_printer, self.printer),
                self.system.dependent_variables,
            ))
        self.ivars, self.idxs, self.th_steps, self.th_sizes = zip(
            *map(
                partial(th_ivar_printer, self.printer),
                self.system.independent_variables,
            ))
        self.dvar_idx = tt.bvector("dvar_idx")
        self.idxs_map = [
            tuple([
                self.system.independent_variables.index(ivar)
                for ivar in dvar.independent_variables
            ]) for dvar in self.system.dependent_variables
        ]

        self.inputs = [*self.dvars, *self.ivars, self.dvar_idx, *self.idxs]
        self.full_inputs = [*self.dvars, *self.ivars]

        self.shapes = [
            list(map(self.printer.doprint, self.system.shapes[dvar]))
            for dvar in self.system.dependent_variables
        ]
        self.shapes = [[
            tt.constant(size, dtype="int32") if isinstance(size, int) else size
            for size in shape
        ] for shape in self.shapes]

        self.sizes = [
            self.printer.doprint(self.system.sizes[dvar])
            for dvar in self.system.dependent_variables
        ]
        self.size = sum(self.sizes)
        self.indices = [th_indices(tt.stack(shape)) for shape in self.shapes]
        self.reshaped_idxs = [
            tt.extra_ops.compress(
                tt.eq(self.dvar_idx, i), tt.stack(self.idxs), axis=1)
            for i in range(self.ndim)
        ]

    def _setup_replacements(self):
        replacement_sizes = {
            size: tt.cast(ivar.size, "int32")
            for ivar, size in zip(self.ivars, self.th_sizes)
        }
        replacement_steps = {
            step: tt.ptp(ivar) / (ivar.size - 1)
            for ivar, step in zip(self.ivars, self.th_steps)
        }
        self._replacement = {**replacement_sizes, **replacement_steps}

    def _pivot_choice(self):
        th_shapes = tt.stacklists(list(self.shapes))
        self.pivot_idx = tt.argsort(tt.sum(th_shapes, axis=0))[::-1]

    def _create_gridinfo(self):
        dvar_info = tt.zeros((self.size, ), dtype="int32")
        idxs_info = tt.zeros((self.ndim, self.size), dtype="int32")
        domains_info = tt.zeros((self.size, ), dtype="int32")

        self._cursors = list(
            accumulate([tt.constant(0, dtype="int32"), *self.sizes]))
        domain_cursor = 0
        self._full_conds = []
        self._full_exprs = []
        self._domains = []

        grids = []
        for i, (sys, cursor, shape, size, indice) in enumerate(
                zip(
                    self.system._system,
                    self._cursors,
                    self.shapes,
                    self.sizes,
                    self.indices,
                )):

            grids.append(indice.reshape(tt.stack([self.ndim, *shape])))

            dvar_info = tt.set_subtensor(dvar_info[cursor:cursor + size],
                                         tt.repeat(tt.constant(i), size))
            idxs_info = tt.set_subtensor(idxs_info[:, cursor:cursor + size],
                                         indice)

            conds, exprs = zip(*sys)
            for j, (cond, expr) in enumerate(zip(conds, exprs)):
                self._full_conds.append(cond)
                self._full_exprs.append(expr)
                self._domains.append(j + domain_cursor)
            th_conds = tt.stacklists(list(map(self.printer.doprint, conds)))
            th_conds_cloned = th.clone(
                th_conds,
                replace={idx: indice[i]
                         for i, idx in enumerate(self.idxs)})

            def get_domain(i):
                return tt.flatten(
                    tt.extra_ops.compress(
                        tt.flatten(th_conds_cloned[:, i]),
                        tt.arange(
                            domain_cursor,
                            domain_cursor + len(conds),
                            dtype="int32"),
                    ))

            domains, _ = th.scan(get_domain, sequences=tt.arange(size))
            domains = tt.flatten(domains)
            domains_info = tt.set_subtensor(domains_info[cursor:cursor + size],
                                            domains)

            domain_cursor += len(conds)

        self.grids = tt.concatenate(
            [idxs_info[self.pivot_idx, :], dvar_info[None, :]], axis=0)

        self._perm_vector = th_lexsort(self.grids[::-1])

        permuted_idxs_info = idxs_info[:, self._perm_vector]
        permuted_dvar_info = dvar_info[None, self._perm_vector]
        permuted_domains_info = domains_info[None, self._perm_vector]
        flatten_idx = tt.arange(self.size, dtype="int32")

        self._gridinfo = tt.concatenate(
            [
                permuted_dvar_info,
                permuted_idxs_info,
                permuted_domains_info,
                flatten_idx[None, :],
            ],
            axis=0,
        ).T

        self.flat_maps = []
        for i, shape in enumerate(self.shapes):
            flat_map = tt.extra_ops.compress(
                tt.eq(self._gridinfo[:, 0], i), self._gridinfo[:, -1], axis=0)
            self.flat_maps.append(flat_map.reshape(shape))

        condlists = [tt.eq(self._gridinfo[:, -2], i) for i in self._domains]
        self.subgrids = [
            tt.extra_ops.compress(condlist, self._gridinfo, axis=0)
            for condlist in condlists
        ]

        self._flat_maps_routine = th.function(self.th_sizes, self.flat_maps)
        self._gridinfo_routine = th.function(self.th_sizes, self._gridinfo)
        self._grids_routine = th.function(self.th_sizes, self.grids)
        self._subgrids_routine = th.function(self.th_sizes, self.subgrids)

        @lru_cache(maxsize=128)
        def compute_flatmaps(*sizes):
            return self._flat_maps_routine(*sizes)

        @lru_cache(maxsize=128)
        def compute_gridinfo(*sizes):
            return np.stack(self._gridinfo_routine(*sizes), axis=0)

        @lru_cache(maxsize=128)
        def compute_grids(*sizes):
            return self._grids_routine(*sizes)

        @lru_cache(maxsize=128)
        def compute_subgrids(*sizes):
            return self._subgrids_routine(*sizes)

        self.compute_flatmaps = compute_flatmaps
        self.compute_gridinfo = compute_gridinfo
        self.compute_grids = compute_grids
        self.compute_subgrids = compute_subgrids

    def _build_decision_trees(self):
        @lru_cache(maxsize=128)
        def build_flat_from_idxs_decision_tree(*sizes):
            gridinfo = self.compute_gridinfo(*sizes)
            X = gridinfo[:, :-2]
            y = gridinfo[:, -1]
            reg = DecisionTreeRegressor()
            reg.fit(X, y)
            return reg

        @lru_cache(maxsize=128)
        def build_idxs_from_flat_decision_tree(*sizes):
            gridinfo = self.compute_gridinfo(*sizes)
            X = gridinfo[:, -1:]
            y = gridinfo[:, :-2]
            reg = DecisionTreeRegressor()
            reg.fit(X, y)
            return reg

        @th.as_op([tt.imatrix, tt.ivector], tt.ivector)
        def th_get_flat_from_idxs(idxs, sizes):
            reg = build_flat_from_idxs_decision_tree(*sizes)
            return reg.predict(idxs).astype("int32")

        @th.as_op([tt.ivector, tt.ivector], tt.imatrix)
        def th_get_idxs_from_flat(flatindex, sizes):
            reg = build_idxs_from_flat_decision_tree(*sizes)
            return reg.predict(flatindex.reshape(-1, 1)).astype("int32")

        self.th_get_flat_from_idxs = th_get_flat_from_idxs
        self.th_get_idxs_from_flat = th_get_idxs_from_flat

        self._input_idxs = tt.imatrix("in_idxs")
        self._input_flat = tt.ivector("in_flat")

        self._get_flat_from_idxs_routine = th.function(
            [self._input_idxs, *self.th_sizes],
            self.th_get_flat_from_idxs(self._input_idxs,
                                       tt.stacklists(self.th_sizes)))
        self._get_idxs_from_flat_routine = th.function(
            [self._input_flat, *self.th_sizes],
            self.th_get_idxs_from_flat(self._input_flat,
                                       tt.stacklists(self.th_sizes)))

        @memory.cache
        def compute_flat_from_idxs(idxs, *sizes):
            return self._get_flat_from_idxs_routine(idxs, *sizes)

        @memory.cache
        def compute_idxs_from_flat(flat, *sizes):
            return self._get_idxs_from_flat_routine(flat, *sizes)

        self.compute_flat_from_idxs = compute_flat_from_idxs
        self.compute_idxs_from_flat = compute_idxs_from_flat

    def _build_U(self):
        self._U = tt.alloc(np.nan, self.size)
        flat_maps = [
            tt.tensor("int32", (False, ) * (self.ndim))
            for i in range(self.ndim)
        ]
        for i, (dvar, shape) in enumerate(zip(self.dvars, self.shapes)):
            flat_map = flat_maps[i]
            cond = tt.eq(self.dvar_idx, i)

            idxs = [tt.extra_ops.compress(cond, idx) for idx in self.idxs]
            dvar = dvar.reshape(shape)[tuple(idxs)]

            self._U = tt.set_subtensor(self._U[flat_map[tuple(idxs)]], dvar)

        self._compute_U = function(
            [*self.inputs, *flat_maps],
            self._U,
            givens=self._replacement,
            on_unused_input="ignore",
            allow_input_downcast=True,
        )
        self._compute_U.trust_input = True

    def _build_idxs(self):
        dvar_idx = tt.concatenate([
            tt.repeat(tt.constant(i, dtype="int32"), size)
            for i, size in enumerate(self.sizes)
        ])

        indices = tt.concatenate(self.indices, axis=1)
        idxs = []
        for i, idx in enumerate(self.idxs):
            idxs.append(indices[tt.constant(i, dtype="int32")])

        self._compute_idxs = function(
            self.th_sizes, [dvar_idx, *idxs], on_unused_input="ignore")

        @lru_cache(maxsize=128)
        def compute_idxs(*sizes):
            return np.stack(self._compute_idxs(*sizes), axis=0)

        self.compute_idxs = compute_idxs

    def _build_evolution_equations(self):
        _subgrids = [tt.imatrix("sub%i" % i) for i in self._domains]
        self.evolution_equations = [
            self.printer.doprint(expr) for expr in self._full_exprs
        ]

        F_ = tt.alloc(
            tt.as_tensor_variable(np.nan),
            (tt.as_tensor_variable(self.size, )))
        for grid, eq in zip(_subgrids, self.evolution_equations):
            eq = th.clone(
                eq,
                replace={
                    self.idxs[i]: grid[:, i + 1]
                    for i in range(self.ndim)
                })
            F_ = tt.set_subtensor(F_[grid[:, -1]], eq)
        F_routine = th.function(
            [*self.inputs, *_subgrids],
            F_,
            allow_input_downcast=True,
            accept_inplace=True,
            on_unused_input="ignore",
            givens=self._replacement)

        def F(fields, parameters={}):
            dvars = fields.data_vars.values()
            sizes = fields.sizes.values()
            ivars = fields.coords.values()
            subgrids = self.compute_subgrids(*sizes)
            return F_routine(*dvars, *ivars, *self.compute_idxs(*sizes),
                             *subgrids)

        self.F = F

    def sort_indexed(self, indexed):
        dvar_idx = [dvar.name
                    for dvar in self.system.dependent_variables].index(
                        str(indexed.args[0]))
        ivars = self.system.dependent_variables[dvar_idx].independent_variables
        idxs = indexed.args[1:]
        idxs = [
            idxs[ivars.index(ivar)] if ivar in ivars else 0
            for ivar in self.system.independent_variables
        ]
        return [dvar_idx, *idxs]

    def _simplify_kron(self, *kron_args):
        kron = KroneckerDelta(*kron_args)
        return kron.subs(
            {ivar.N: oo
             for ivar in self.system.independent_variables})

    def _build_jacobian(self):
        self._full_jacs = []
        self._full_jacs_cols = []
        for expr in self._full_exprs:
            wrts = expr.atoms(Indexed)
            wrts, grids = wrts, list(map(self.sort_indexed, wrts))
            grids = [[
                tt.constant(int(idx), dtype="int32") if isinstance(
                    idx, (int, Number)) else self.printer.doprint(idx)
                for idx in grid
            ] for grid in grids]
            self._full_jacs_cols.append(grids)
            diffs = [
                expr.diff(wrt).replace(KroneckerDelta,
                                       self._simplify_kron).n().simplify()
                for wrt in wrts
            ]
            self._full_jacs.append(
                [self.printer.doprint(diff) for diff in diffs])

        rows = []
        cols = []
        data = []

        _subgrids = [tt.imatrix("sub%i" % i) for i in self._domains]

        for grid, jacs, jac_cols in zip(_subgrids, self._full_jacs,
                                        self._full_jacs_cols):
            for col_func, jac in zip(jac_cols, jacs):
                J_ = tt.zeros((grid.shape[0], ))
                cols_ = tt.zeros((grid.shape[0], self.ndim + 1), dtype="int32")
                jac = th.clone(
                    jac,
                    replace={
                        self.idxs[i]: grid[:, i + 1]
                        for i in range(self.ndim)
                    })
                cols_idxs = th.clone(
                    col_func,
                    replace={
                        self.idxs[i]: grid[:, i + 1]
                        for i in range(self.ndim)
                    })
                jac = tt.set_subtensor(J_[:], jac)
                for i, col in enumerate(cols_idxs):
                    cols_ = tt.set_subtensor(cols_[:, i], col)
                cols_idxs = tt.unbroadcast(cols_, 0)
                flat_cols = self.th_get_flat_from_idxs(cols_idxs,
                                                       tt.stacklists(
                                                           self.th_sizes))
                rows.append(grid[:, -1].reshape((-1, )))
                cols.append(flat_cols.reshape((-1, )))
                data.append(jac.reshape((-1, )))

        data = tt.concatenate(data)
        rows = tt.concatenate(rows)
        cols = tt.concatenate(cols)

        data_ = tt.dvector()
        rows_ = tt.ivector()
        cols_ = tt.ivector()

        permutation = tt.lvector("perm")
        pdata = data_[permutation]
        prows = rows_[permutation]
        pcols = cols_[permutation]

        count = tt.zeros((self.size + 1, ), dtype=int)
        uq, cnt = tt.extra_ops.Unique(False, False, True)(pcols)
        count = tt.set_subtensor(count[uq + 1], cnt)

        indptr = tt.cumsum(count)
        shape = tt.stack([self.size, self.size])

        Jrows_routine = th.function(
            [*self.th_sizes, *_subgrids],
            rows,
            on_unused_input="ignore",
            allow_input_downcast=True)
        Jcols_routine = th.function(
            [*self.th_sizes, *_subgrids],
            cols,
            on_unused_input="ignore",
            allow_input_downcast=True)
        Jdata_routine = th.function(
            [*self.inputs, *_subgrids],
            data,
            givens=self._replacement,
            on_unused_input="ignore",
            allow_input_downcast=True)

        coo_to_csr = th.function(
            [data_, rows_, cols_, *self.th_sizes, permutation],
            [pdata, prows, indptr, shape])

        @lru_cache(maxsize=128)
        def compute_J_coords(*sizes):
            subgrids = self.compute_subgrids(*sizes)
            rows, cols = Jrows_routine(*sizes, *subgrids), Jcols_routine(
                *sizes, *subgrids)
            perm = np.argsort(cols)
            return rows, cols, perm

        def compute_J_data(ivars, dvars):
            sizes = [ivar.size for ivar in ivars]
            return Jdata_routine(*dvars, *ivars, *self.compute_idxs(*sizes),
                                 *self.compute_subgrids(*sizes))

        def J(fields, parameters={}):
            dvars = [dvar.values for dvar in fields.data_vars.values()]
            sizes = [size for size in fields.sizes.values()]
            ivars = [ivar.values for ivar in fields.coords.values()]

            data = compute_J_data(ivars, dvars)
            rows, cols, perm = compute_J_coords(*sizes)
            data, indices, indptr, shape = coo_to_csr(data, rows, cols, *sizes,
                                                      perm)
            return csc_matrix((data, indices, indptr), shape)

        self.J = J

    def __attrs_post_init__(self):
        self.printer = self.Printer(self.system)
        self._convert_inputs()
        self._setup_replacements()
        self._pivot_choice()
        self._create_gridinfo()
        self._build_idxs()
        self._build_decision_trees()
        self._build_evolution_equations()
        self._build_jacobian()
