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
from joblib.parallel import Parallel, delayed
from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.theanocode import (
    TheanoPrinter,
    dim_handling,
    mapping,
    theano_function,
)
from triflow.core.system import PDESys

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

cachedir = mkdtemp()
memory = Memory(location=cachedir)


def np_and(*args):
    return np.logical_and.reduce(args)


def np_depvar_printer(printer, dvar):
    return printer.doprint(dvar.discrete)


def np_ivar_printer(printer, ivar):
    return (
        printer.doprint(ivar.discrete),
        printer.doprint(Symbol(str(ivar.idx))),
        printer.doprint(ivar.step),
        printer.doprint(ivar.N),
    )


@attr.s
class NumpyCompiler:
    Printer = NumPyPrinter
    system = attr.ib(type=PDESys)
    n_jobs = attr.ib(-1, type=int)

    def _convert_inputs(self):
        self.ndim = len(self.system.independent_variables)
        self.dvars = list(
            map(
                partial(np_depvar_printer, self.printer),
                self.system.dependent_variables,
            )
        )
        self.pars = list(
            map(partial(np_depvar_printer, self.printer), self.system.parameters)
        )
        self.ivars, self.idxs, self.steps, self.sizes = zip(
            *map(
                partial(np_ivar_printer, self.printer),
                self.system.independent_variables,
            )
        )
        self.t = self.printer.doprint(self.system._t)
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
            list(map(self.printer.doprint, self.system.shapes[dvar]))
            for dvar in self.system.dependent_variables
        ]

        self.inputs = [
            self.t,
            *self.dvars,
            *self.pars,
            *self.ivars,
            *self.idxs,
            *self.steps,
            *self.sizes,
        ]
        self.inputs_cond = [*self.idxs, *self.sizes]

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

        def get_flat_from_idxs(idxs, sizes):
            reg = build_flat_from_idxs_decision_tree(*sizes)
            return reg.predict(idxs).astype("int32")

        def get_idxs_from_flat(flatindex, sizes):
            reg = build_idxs_from_flat_decision_tree(*sizes)
            return reg.predict(flatindex.reshape(-1, 1)).astype("int32")

        self.get_flat_from_idxs = get_flat_from_idxs
        self.get_idxs_from_flat = get_idxs_from_flat

        @memory.cache
        def compute_flat_from_idxs(idxs, *sizes):
            return self.get_flat_from_idxs(idxs, *sizes)

        @memory.cache
        def compute_idxs_from_flat(flat, *sizes):
            return self.get_idxs_from_flat(flat, *sizes)

        self.compute_flat_from_idxs = compute_flat_from_idxs
        self.compute_idxs_from_flat = compute_idxs_from_flat

    def compute_pivot_idx(self, shapes):
        if np.alltrue(shapes == shapes[0]):
            return np.arange(self.ndim)
        return np.argsort(np.sum(shapes, axis=0))[::-1]

    def _create_base_routines(self):
        # lambdify the shape computation for each dep. var.
        self._lambda_shapes = [
            lambdify(self.sizes, shape, modules="numpy") for shape in self.shapes
        ]

        # lambdify the size computation for each dep. var. (shape product)
        self._lambda_sizes = [
            lambdify(self.sizes, self.system.sizes[dvar], modules="numpy")
            for dvar in self.system.dependent_variables
        ]

        @lru_cache(maxsize=128)
        def compute_shapes(*sizes):
            """
            compute the shapes of all the model fields.
            """
            return np.stack(
                [lambda_shape(*sizes) for lambda_shape in self._lambda_shapes]
            )

        @lru_cache(maxsize=128)
        def compute_steps(sizes, lenghts):
            """
            compute the stepsize of all the independent variables.
            """
            return [lenght / (size + 1) for lenght, size in zip(lenghts, sizes)]

        @lru_cache(maxsize=128)
        def compute_sizes(*sizes):
            """
            compute the sizes of all the model fields.
            """
            return np.stack([lambda_size(*sizes) for lambda_size in self._lambda_sizes])

        @lru_cache(maxsize=128)
        def compute_indices(*sizes):
            """
            compute grid indices of all the model fields.
            """
            shapes = compute_shapes(*sizes)
            return [np.indices(np.stack(shape)) for shape in shapes]

        self._compute_shapes = compute_shapes
        self._compute_steps = compute_steps
        self._compute_sizes = compute_sizes
        self._compute_indices = compute_indices

        self._lambda_conds = []
        self._lambda_domain = []
        self._lambda_exprs = []
        domain_cursor = 0
        for sys in self.system._system:
            conds, _ = zip(*sys)
            lambda_cond = lambdify(
                self.inputs_cond,
                Matrix(conds),
                modules=[dict(logical_and=np_and), "numpy"],
            )
            self._lambda_conds.append(lambda_cond)

            def get_domain(idxs, sizes, cursor=0):
                return np.select(
                    lambda_cond(*idxs, *sizes), np.arange(cursor, cursor + len(conds))
                ).squeeze()

            local_domain_computation = partial(get_domain, cursor=domain_cursor)
            self._lambda_domain.append(local_domain_computation)
            domain_cursor = domain_cursor + len(conds)

        @lru_cache(maxsize=128)
        def compute_domains(*sizes):
            """
            compute grid indices of all the model fields.
            """
            indices = compute_indices(*sizes)
            return np.stack(
                [
                    lambda_domain(indice, sizes)
                    for indice, lambda_domain in zip(indices, self._lambda_domain)
                ]
            )

        self._compute_domains = compute_domains

    def _build_grid_routines(self):
        @lru_cache(maxsize=128)
        def compute_gridinfo(*sizes):
            system_sizes = self._compute_sizes(*sizes)
            system_size = sum(system_sizes)
            shapes = self._compute_shapes(*sizes)
            indices = self._compute_indices(*sizes)
            domains = self._compute_domains(*sizes)
            ndim = self.ndim

            pivot_idx = self.compute_pivot_idx(shapes)
            cursors = list(accumulate([0, *system_sizes]))

            dvar_info = np.zeros((system_size,), dtype="int32")
            idxs_info = np.zeros((ndim, system_size), dtype="int32")
            domains_info = np.zeros((system_size,), dtype="int32")

            for i, (cursor, size, indice, domain) in enumerate(
                zip(cursors, system_sizes, indices, domains)
            ):
                dvar_info[cursor : cursor + size] = np.repeat(i, size)
                idxs_info[:, cursor : cursor + size] = indice.reshape((-1, size))
                domains_info[cursor : cursor + size] = domain.flatten()

            full_grid = np.concatenate(
                [idxs_info[pivot_idx, :], dvar_info[None, :]], axis=0
            )

            perm_vector = np.lexsort(full_grid[::-1])

            permuted_idxs_info = idxs_info[:, perm_vector]
            permuted_dvar_info = dvar_info[None, perm_vector]
            permuted_domains_info = domains_info[None, perm_vector]
            flatten_idx = np.arange(system_size, dtype="int32")

            gridinfo = np.concatenate(
                [
                    permuted_dvar_info,
                    permuted_idxs_info,
                    permuted_domains_info,
                    flatten_idx[None, :],
                ],
                axis=0,
            ).T

            return gridinfo

        self.compute_gridinfo = compute_gridinfo

        @lru_cache(maxsize=128)
        def compute_flat_maps(*sizes):
            shapes = self._compute_shapes(*sizes)
            gridinfo = compute_gridinfo(*sizes)
            flatmaps = [
                np.compress(gridinfo[:, 0] == i, gridinfo[:, -1], axis=0).reshape(shape)
                for i, shape in enumerate(shapes)
            ]
            return flatmaps

        self.compute_flat_maps = compute_flat_maps

        @lru_cache(maxsize=128)
        def compute_subgrids(*sizes):
            gridinfo = self.compute_gridinfo(*sizes)
            domains = sorted(set(self._compute_domains(*sizes).flatten().tolist()))
            condlists = [gridinfo[:, -2] == i for i in domains]
            subgrids = [
                np.compress(condlist, gridinfo, axis=0) for condlist in condlists
            ]
            return subgrids

        self.compute_subgrids = compute_subgrids

        @lru_cache(maxsize=128)
        def compute_flat_to_dvars(*sizes):
            gridinfo = self.compute_gridinfo(*sizes)
            system_sizes = self._compute_sizes(*sizes)
            idxs_grids = [
                self.get_flat_from_idxs(
                    np.compress(gridinfo[:, 0] == i, gridinfo[:, :-2], axis=0), sizes
                )
                for i in range(len(self.dvars))
            ]

            return idxs_grids

        self.compute_flat_to_dvars = compute_flat_to_dvars

        @lru_cache(maxsize=128)
        def compute_dvars_to_flat(*sizes):
            system_sizes = self._compute_sizes(*sizes)
            system_size = sum(system_sizes)
            ptrs = self.get_idxs_from_flat(np.arange(system_size, dtype="int32"), sizes)
            return ptrs

        self.compute_dvars_to_flat = compute_dvars_to_flat

    def _build_idxs(self):
        @lru_cache(maxsize=128)
        def compute_idxs(*sizes):
            system_sizes = self._compute_sizes(*sizes)
            indices = self._compute_indices(*sizes)
            indices = [
                indice.reshape(-1, size) for indice, size in zip(indices, system_sizes)
            ]
            dvar_idx = np.concatenate(
                [np.repeat(i, size) for i, size in enumerate(system_sizes)]
            )
            idxs = np.concatenate(
                [indices[i] for i, size in enumerate(system_sizes)], axis=1
            )
            return np.stack([dvar_idx, *idxs], axis=0)

        self.compute_idxs = compute_idxs

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
            subgrids = self.compute_subgrids(*sizes)
            steps = self._compute_steps(
                tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
            )
            system_sizes = self._compute_sizes(*sizes)
            system_size = sum(system_sizes)
            Fs = []
            for grid, eq in zip(subgrids, evolution_equations):
                Fs.append(
                    delayed(eq)(
                        t, *dvars, *pars, *ivars, *grid[:, 1:-2].T, *steps, *sizes
                    )
                )
            Fs = Parallel(n_jobs=self.n_jobs, backend="threading")(Fs)
            F = np.empty(system_size)
            for grid, F_i in zip(subgrids, Fs):
                F[grid[:, -1]] = F_i
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

    def _build_flattener(self):
        def U_routine(dvars, sizes):
            ptrs = self.compute_dvars_to_flat(*sizes)
            return np.stack(dvars, axis=0)[
                tuple([ptrs.T[i] for i in range(self.ndim + 1)])
            ]

        def U_from_fields(fields, t=0):
            dvars = [
                fields[varname]
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            sizes = [
                fields[varname].size
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]

            return U_routine(dvars, sizes)

        self.U_from_fields = U_from_fields

        def fields_routine(U, sizes):
            # FIXME better naming
            shapeinfos = self._compute_shapes(*sizes)
            pivot_idx = self.compute_pivot_idx(shapeinfos)
            idxs_grids = self.compute_flat_to_dvars(*sizes)

            shapes = [
                np.take(shape, np.take(pivot_idx, idx_map))
                for idx_map, shape in zip(self.idxs_map, shapeinfos)
            ]

            return [
                U[grid].reshape(shape)
                for grid, shape, shapeinfo in zip(idxs_grids, shapes, shapeinfos)
            ]

        def fields_from_U(U, fields, t=None):
            varnames = [dvar.name for dvar in self.system.dependent_variables]
            fields = fields.copy()

            ivars = [
                fields[varname]
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]
            shapes = self._compute_shapes(*sizes)
            pivots = self.compute_pivot_idx(shapes)
            dvars = fields_routine(U, sizes)
            for varname, dvar, ivars in zip(
                varnames,
                dvars,
                [
                    dvar.independent_variables
                    for dvar in self.system.dependent_variables
                ],
            ):
                fields[varname] = [ivars[i].name for i in pivots], dvar
            return fields

        self.fields_from_U = fields_from_U

    def _build_jacobian(self):
        self._full_jacs = []
        self._full_jacs_cols = []
        for expr in self._full_exprs:
            wrts = list(filter(self.filter_dvar_indexed, expr.atoms(Indexed)))
            wrts, grids = wrts, list(map(self.sort_indexed, wrts))
            self._full_jacs_cols.append(
                [
                    lambdify(
                        self.inputs_cond, self.printer.doprint(grid), modules="numpy"
                    )
                    for grid in grids
                ]
            )
            diffs = [
                expr.diff(wrt).replace(KroneckerDelta, self._simplify_kron).n()
                for wrt in wrts
            ]
            self._full_jacs.append(
                [
                    lambdify(self.inputs, self.printer.doprint(diff), modules="numpy")
                    for diff in diffs
                ]
            )

        def compute_jacobian_values(t, dvars, pars, ivars, sizes):
            subgrids = self.compute_subgrids(*sizes)
            steps = self._compute_steps(
                tuple(sizes), tuple([ivar.ptp() for ivar in ivars])
            )
            data_size = sum(
                [
                    subgrid.shape[0] * len(jac_col)
                    for subgrid, jac_col in zip(subgrids, self._full_jacs_cols)
                ]
            )
            data = np.zeros(data_size)

            cursor = 0
            for grid, jacs in zip(subgrids, self._full_jacs):
                for jac_func in jacs:
                    next_cursor = cursor + grid.size
                    jac = jac_func(
                        t, *dvars, *pars, *ivars, *grid[:, 1:-2].T, *steps, *sizes
                    )
                    data[cursor:next_cursor] = jac
                    cursor = next_cursor
            return data

        @lru_cache(maxsize=128)
        def compute_jacobian_coordinates(*sizes):
            subgrids = self.compute_subgrids(*sizes)
            system_sizes = self._compute_sizes(*sizes)
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

                    flat_cols = self.get_flat_from_idxs(cols.T, sizes)

                    rows_list.extend(grid[:, -1].reshape((-1,)))
                    cols_list.extend(flat_cols.reshape((-1,)))
            rows = np.array(rows_list)
            cols = np.array(cols_list)

            perm = np.argsort(cols)
            rows = rows[perm]
            cols = cols[perm]
            count = np.zeros((system_size + 1), dtype="int32")
            uq, cnt = np.unique(cols, False, False, True)
            count[uq + 1] = cnt
            indptr = np.cumsum(count)
            return rows, indptr, perm, (system_size, system_size)

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
            rows, indptr, perm, shape = compute_jacobian_coordinates(*sizes)

            return csc_matrix((data[perm], rows, indptr), shape=shape)

        self.J = J

    def __attrs_post_init__(self):
        logging.info("numpy compiler: init printer...")
        self.printer = self.Printer()
        logging.info("numpy compiler: convert_inputs...")
        self._convert_inputs()
        logging.info("numpy compiler: build_decision_trees...")
        self._build_decision_trees()
        logging.info("numpy compiler: create_base_routines...")
        self._create_base_routines()
        logging.info("numpy compiler: create_grid_routines...")
        self._build_grid_routines()
        logging.info("numpy compiler: build_flattener...")
        self._build_flattener()
        logging.info("numpy compiler: build_idxs...")
        self._build_idxs()
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
