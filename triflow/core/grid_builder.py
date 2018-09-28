#!/usr/bin/env python
# coding=utf-8

import logging
from functools import lru_cache, partial
from itertools import accumulate
from tempfile import mkdtemp

import attr
import numpy as np
from joblib import Memory
from sklearn.tree import DecisionTreeRegressor
from sympy import lambdify, Matrix
from .system import PDESys


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

cachedir = mkdtemp()
memory = Memory(location=cachedir)


@attr.s
class GridBuilder:
    system = attr.ib(type=PDESys)

    def _convert_inputs(self):
        self.ndim = len(self.system.independent_variables)
        self.dvars = [dvar.symbol for dvar in self.system.dependent_variables]
        self.sizes = [ivar.N for ivar in self.system.independent_variables]
        self.idxs = [ivar.idx for ivar in self.system.independent_variables]
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
            return [lenght / (size - 1) for lenght, size in zip(lenghts, sizes)]

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

        self.compute_shapes = compute_shapes
        self.compute_steps = compute_steps
        self.compute_sizes = compute_sizes
        self.compute_indices = compute_indices

        self._lambda_conds = []
        self._lambda_domain = []
        self._lambda_exprs = []

        for sys in self.system._system:
            conds, _ = zip(*sys)
            lambda_cond = lambdify(self.inputs_cond, Matrix(conds), modules=["numpy"])
            self._lambda_conds.append(lambda_cond)

        @lru_cache(maxsize=128)
        def compute_domains(*sizes):
            """
            compute grid indices of all the model fields.
            """
            indices = self.compute_indices(*sizes)
            cursor = 0
            domains = []
            for lambda_cond, indice in zip(self._lambda_conds, indices):
                cond_grid = lambda_cond(*indice, *sizes)
                domain_grid = np.select(
                    cond_grid, np.arange(cursor, cursor + cond_grid.shape[0])
                ).squeeze()
                cursor = domain_grid.max() + 1
                domains.append(domain_grid)
            return domains

        self.compute_domains = compute_domains

    def _build_grid_routines(self):
        @lru_cache(maxsize=128)
        def compute_gridinfo(*sizes):
            system_sizes = self.compute_sizes(*sizes)
            system_size = sum(system_sizes)
            shapes = self.compute_shapes(*sizes)
            indices = self.compute_indices(*sizes)
            domains = self.compute_domains(*sizes)
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
            shapes = self.compute_shapes(*sizes)
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
            domains = sorted(
                set(
                    np.concatenate(
                        [domain.flatten() for domain in self.compute_domains(*sizes)]
                    ).tolist()
                )
            )
            condlists = [gridinfo[:, -2] == i for i in domains]
            subgrids = [
                np.compress(condlist, gridinfo, axis=0) for condlist in condlists
            ]
            return subgrids

        self.compute_subgrids = compute_subgrids

        @lru_cache(maxsize=128)
        def compute_flat_to_dvars(*sizes):
            gridinfo = self.compute_gridinfo(*sizes)
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
            system_sizes = self.compute_sizes(*sizes)
            system_size = sum(system_sizes)
            ptrs = self.get_idxs_from_flat(np.arange(system_size, dtype="int32"), sizes)
            return ptrs

        self.compute_dvars_to_flat = compute_dvars_to_flat

    def _build_idxs(self):
        @lru_cache(maxsize=128)
        def compute_idxs(*sizes):
            system_sizes = self.compute_sizes(*sizes)
            indices = self.compute_indices(*sizes)
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

    def _build_flattener(self):
        @lru_cache(maxsize=128)
        def get_relevant_ptrs(i, *sizes):
            system_sizes = self.compute_sizes(*sizes)
            system_size = sum(system_sizes)
            shapeinfos = self.compute_shapes(*sizes)
            pivot_idx = self.compute_pivot_idx(shapeinfos)
            idxs = np.arange(system_size)
            ptrs = self.compute_dvars_to_flat(*sizes)
            relevant_idxs = np.extract(ptrs[:, 0] == i, idxs)
            relevant_ptrs = ptrs[relevant_idxs, 1:].T[pivot_idx]
            return relevant_idxs, tuple(relevant_ptrs)

        def U_routine(dvars, sizes):
            system_sizes = self.compute_sizes(*sizes)
            system_size = sum(system_sizes)
            U = np.empty(system_size)

            for i, dvar in enumerate(dvars):
                relevant_idxs, relevant_ptrs = get_relevant_ptrs(i, *sizes)
                U[relevant_idxs] = dvar[relevant_ptrs]
            return U

        def U_from_fields(fields, t=0):
            dvars = [
                fields[varname]
                for varname in [dvar.name for dvar in self.system.dependent_variables]
            ]
            sizes = [
                fields[varname].size
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            shapeinfos = self.compute_shapes(*sizes)
            pivot_idx = self.compute_pivot_idx(shapeinfos)
            ivars = list(
                map(str, np.array(self.system.independent_variables)[pivot_idx])
            )

            dvars = [
                dvar.expand_dims(set(fields.dims).difference(set(dvar.dims)))
                .transpose(*ivars)
                .values
                for dvar in dvars
            ]

            return U_routine(dvars, sizes)

        self._U_routine = U_routine
        self.U_from_fields = U_from_fields

        def fields_routine(U, sizes):
            shapes = self.compute_shapes(*sizes)
            idxs_grids = self.compute_flat_to_dvars(*sizes)
            pivots = self.compute_pivot_idx(shapes)

            return [
                U[grid].reshape(shape[pivots])
                for grid, shape in zip(idxs_grids, shapes)
            ]

        def fields_from_U(U, fields, t=None):
            varnames = [dvar.name for dvar in self.system.dependent_variables]
            fields = fields.copy()

            ivars = [
                fields[varname]
                for varname in [ivar.name for ivar in self.system.independent_variables]
            ]
            sizes = [ivar.size for ivar in ivars]
            shapes = self.compute_shapes(*sizes)
            pivots = self.compute_pivot_idx(shapes)
            dvars = fields_routine(U, sizes)
            sys_ivars = self.system.independent_variables
            for varname, dvar, ivars in zip(
                varnames,
                dvars,
                [
                    dvar.independent_variables
                    for dvar in self.system.dependent_variables
                ],
            ):
                coords = [sys_ivars[i].name for i in pivots if sys_ivars[i] in ivars]
                fields[varname] = coords, dvar.squeeze()
            return fields.transpose(*[ivar.name for ivar in sys_ivars])

        self.fields_from_U = fields_from_U

    def __attrs_post_init__(self):
        logging.info("grid builder: convert_inputs...")
        self._convert_inputs()
        logging.info("grid builder: build_decision_trees...")
        self._build_decision_trees()
        logging.info("grid builder: create_base_routines...")
        self._create_base_routines()
        logging.info("grid builder: create_grid_routines...")
        self._build_grid_routines()
        logging.info("grid builder: build_flattener...")
        self._build_flattener()
        logging.info("grid builder: build_idxs...")
        self._build_idxs()
