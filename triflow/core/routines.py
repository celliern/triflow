#!/usr/bin/env python
# coding=utf8

from pickle import loads, dumps
import numpy as np
import scipy.sparse as sps
from sympy.printing.theanocode import theano_function
from toolz import memoize


@memoize
def get_indices(N, window_range, nvar, mode, sparse_indices):
    i = np.arange(N)[:, np.newaxis]
    idx = np.arange(N * nvar).reshape((nvar, N), order='F')
    idx = np.pad(idx, ((0, 0),
                       (int((window_range - 1) / 2),
                        int((window_range - 1) / 2))),
                 mode=mode).flatten('F')
    sparse_indices = [list(sparse_indices)]
    unknowns_idx = np.arange(window_range *
                             nvar) + i * nvar
    rows = np.tile(np.arange(nvar),
                   window_range * nvar) + i * nvar
    cols = np.repeat(np.arange(window_range * nvar),
                     nvar) + i * nvar
    rows = rows[:, sparse_indices]
    cols = idx[cols][:, sparse_indices]

    return idx, unknowns_idx, rows, cols


def reduce_routine(matrix, args, window_range, pars,
                   class_routine, dumped_routine):
    model = class_routine(matrix, args, window_range, pars, reduced=True)
    model.ufunc = loads(dumped_routine)
    return model


class ModelRoutine:
    def __init__(self, matrix, args, window_range, pars, reduced=False):
        self.pars = pars
        self.window_range = window_range
        self.matrix = matrix
        self.args = args
        if not reduced:
            self.make_ufuncs()

    def make_ufuncs(self):
        # self.routine_dir = Path(tempdir(prefix='triflow_routines_'))
        self.ufunc = theano_function(inputs=self.args,
                                     outputs=self.matrix.flatten().tolist(),
                                     on_unused_input='ignore',
                                     dim=1)

    def __repr__(self):
        return self.matrix.__repr__()

    def __reduce__(self):
        return reduce_routine, (self.matrix, self.args,
                                self.window_range, self.pars,
                                self.__class__, dumps(self.ufunc))


class H_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        N = fields.size
        middle_point = int((self.window_range - 1) / 2)
        mode = 'wrap' if pars.get('periodic', False) else 'edge'
        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.window_range)],
                               axis=1).T
        pargs = [pars[key] + fields.x.squeeze() * 0
                 for key
                 in self.pars] + [pars['dx'] + fields.x.squeeze() * 0]
        F = np.array(self.ufunc((*uargs.tolist() + pargs))).flatten()
        return F


class F_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        N = fields.size
        middle_point = int((self.window_range - 1) / 2)
        mode = 'wrap' if pars.get('periodic', False) else 'edge'
        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.window_range)],
                               axis=1).T
        pargs = [pars[key] + fields.x.squeeze() * 0
                 for key
                 in self.pars] + [pars['dx'] + fields.x.squeeze() * 0]
        F = np.array(self.ufunc((*uargs.tolist() + pargs))).flatten('F')
        return F

    def diff_approx(self, fields, pars, eps=1E-8):
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
            Fplus = self(fields_plus, pars)
            Fmoins = self(fields_moins, pars)
            J[i] = (Fplus - Fmoins) / (2 * eps)

        return J.T


class J_Routine(ModelRoutine):
    def __init__(self, matrix, args, window_range, pars, reduced=False):
        super().__init__(matrix, args, window_range, pars, reduced=False)
        self.matrix = self.matrix.flatten('F')
        self.sparse_indices = np.where(matrix != 0)

        self.matrix = self.matrix[self.sparse_indices]

    def make_ufuncs(self):
        self.ufunc = theano_function(inputs=self.args,
                                     outputs=self.matrix.tolist(),
                                     on_unused_input='ignore',
                                     dim=1)

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
                                   mode,
                                   tuple(self.sparse_indices[0]))

        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.window_range)],
                               axis=1).T
        pargs = [pars[key] + fields.x.squeeze() * 0
                 for key
                 in self.pars] + [pars['dx'] + fields.x.squeeze() * 0]

        J = self.ufunc((*uargs.tolist() + pargs))
        J = np.vstack(J)
        J = sps.csr_matrix((J.T.flatten(),
                            (rows.flatten(),
                             cols.flatten())),
                           (N * nvar,
                            N * nvar))
        return J if sparse else J.todense()
