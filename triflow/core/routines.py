#!/usr/bin/env python
# coding=utf8

from functools import partial

import tarfile
import importlib.util as impu
import numpy as np
import scipy.sparse as sps
from path import tempdir, Path
from sympy.utilities.autowrap import ufuncify
from toolz import memoize
import io


@memoize
def get_indices(N, window_range, nvar, mode):
    i = np.arange(N)[:, np.newaxis]
    idx = np.arange(N * nvar).reshape((nvar, N), order='F')
    idx = np.pad(idx, ((0, 0),
                       (int((window_range - 1) / 2),
                        int((window_range - 1) / 2))),
                 mode=mode).flatten('F')
    unknowns_idx = np.arange(window_range *
                             nvar) + i * nvar
    rows = np.tile(np.arange(nvar),
                   window_range * nvar) + i * nvar
    cols = np.repeat(np.arange(window_range * nvar),
                     nvar) + i * nvar
    rows = rows
    cols = idx[cols]
    return idx, unknowns_idx, rows, cols


def reduce_routine(BaseClass, matrix, args, window_range, pars, stream, names):
    tmp_dir = Path(tempdir(prefix='triflow_routines_'))
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode='r|bz2') as tar:
        tar.extractall(tmp_dir)
    routine = BaseClass(matrix, args, window_range, pars, reduced=True)
    routine.load_ufuncs(tmp_dir, names)
    return routine


class ModelRoutine:
    def __init__(self, matrix, args, window_range, pars, reduced=False):
        self.pars = pars
        self.window_range = window_range
        self.matrix = matrix
        self.args = args
        if not reduced:
            self.make_ufuncs()

    def make_ufuncs(self):
        self.routine_dir = Path(tempdir(prefix='triflow_routines_'))
        self.ufuncs = np.array(list(map(partial(ufuncify, self.args,
                                                tempdir=self.routine_dir),
                                        np.array(self.matrix)
                                        .flatten(order='F'))))
        self.ufunc_names = [func.__name__ for func in self.ufuncs]

    def load_ufuncs(self, tmp_dir, names):
        self.routine_dir = tmp_dir
        self.ufunc_names = names
        self.ufuncs = []
        for name in names:
            file = self.routine_dir.files(f'{name}*.so').pop()
            spec = impu.spec_from_file_location(name,
                                                file)
            module = impu.module_from_spec(spec)
            self.ufuncs.append(module.autofunc)

    def __repr__(self):
        return self.matrix.__repr__()

    def __reduce__(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w|bz2') as tar:
            for name in self.ufunc_names:
                file = self.routine_dir.files(f'{name}*.so').pop()
                tar.add(file, arcname=file.basename())

        return reduce_routine, (self.__class__,
                                self.matrix, self.args, self.window_range,
                                self.pars, stream, self.ufunc_names)

    def __del__(self):
        try:
            self.routine_dir.rmtree_p()
        except NameError:
            pass


class H_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        N = fields.size
        middle_point = int((self.window_range - 1) / 2)
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if pars.get('periodic', False) else 'edge'
        F = np.zeros((1, N))
        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs = [pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufuncs):
            F[i] = ufunc((*uargs.tolist() + pargs)).squeeze()
        return F.flatten('F')


class F_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        nvar, N = len(fields.vars), fields.size
        middle_point = int((self.window_range - 1) / 2)
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if pars.get('periodic', False) else 'edge'
        F = np.zeros((nvar, N))
        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.window_range)],
                               axis=1).T
        pargs = [pars[key] for key in self.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufuncs):
            F[i, :] = ufunc((*uargs.tolist() + pargs))
        return F.flatten('F')

    def num_approx(self, fields, pars, eps=1E-8):
        nvar, N = len(fields.vars), fields.size
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = pars['dx']
        J = np.zeros((N * nvar, N * nvar))
        indices = np.indices(fields.uarray.shape)
        for i, (var_index, node_index) in enumerate(zip(*map(np.ravel,
                                                             indices))):
            fields_plus = fields.copy()
            fields_plus.uarray[var_index, node_index] += eps
            fields_moins = fields.copy()
            fields_moins.uarray[var_index, node_index] -= eps
            Fplus = self.model.F(fields_plus, pars)
            Fmoins = self.model.F(fields_moins, pars)
            J[i] = (Fplus - Fmoins) / (2 * eps)

        return J.T


class J_Routine(ModelRoutine):
    def __call__(self, fields, pars, sparse=True):
        nvar, N = len(fields.vars), fields.size
        middle_point = int((self.window_range - 1) / 2)
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if pars.get('periodic', False) else 'edge'
        J = np.zeros((self.window_range * nvar ** 2, N))

        (idx, unknowns_idx,
         rows, cols) = get_indices(N,
                                   self.window_range,
                                   nvar,
                                   mode)

        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.window_range)],
                               axis=1).T
        pargs = [pars[key] for key in self.pars] + [pars['dx']]
        for i, ujacob in enumerate(self.ufuncs):
            Ji = ujacob((*uargs.tolist() + pargs))
            J[i] = Ji
        J = sps.csr_matrix((J.T.flatten(),
                            (rows.flatten(),
                             cols.flatten())),
                           (N * nvar,
                            N * nvar))
        return J if sparse else J.todense()
