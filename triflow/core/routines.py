#!/usr/bin/env python
# coding=utf8

from path import tempdir
import scipy.sparse as sps
import numpy as np
from sympy.utilities.autowrap import ufuncify
from toolz import memoize
from functools import partial


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


def reduce_routine(matrix, args, model):
    routine = ModelRoutine(matrix, args, model, reduced=True)
    return routine


class ModelRoutine:
    def __init__(self, matrix, model, reduced=False):
        self.routine_dir = tempdir()
        self.matrix = matrix
        self.args = model.sargs
        self.model = model
        if not reduced:
            self.make_ufunc()
        else:
            self.load_ufunc()

    def make_ufunc(self):
        self.ufunc = np.array(list(map(partial(ufuncify, self.args,
                                               tempdir=self.routine_dir),
                                       np.array(self.matrix)
                                       .flatten(order='F'))))
        self.ufunc_names = [func.__name__ for func in self.ufunc]

    def load_ufunc(self):
        pass

    def __repr__(self):
        return self.matrix.__repr__()

    def __reduce__(self):
        return reduce_routine(self.matrix, self.args,
                              self.model, reduced=True)


class H_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        N=fields.size
        middle_point=int((self.model.window_range - 1) / 2)
        fpars={key: pars[key] for key in self.model.pars}
        fpars['dx']=pars['dx']
        mode='wrap' if self.model.periodic else 'edge'
        F=np.zeros((1, N))
        unknowns=np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs=np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs=[pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufunc):
            F[i]=ufunc((*uargs.tolist() + pargs)).squeeze()
        return F.flatten('F')


class F_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        nvar, N=len(fields.vars), fields.size
        middle_point=int((self.model.window_range - 1) / 2)
        fpars={key: pars[key] for key in self.model.pars}
        fpars['dx']=pars['dx']
        mode='wrap' if self.model.periodic else 'edge'
        F=np.zeros((nvar, N))
        unknowns=np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs=np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs=[pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufunc):
            F[i, :]=ufunc((*uargs.tolist() + pargs))
        return F.flatten('F')


class J_Routine(ModelRoutine):
    def __call__(self, fields, pars, sparse=True):
        nvar, N=len(fields.vars), fields.size
        middle_point=int((self.model.window_range - 1) / 2)
        fpars={key: pars[key] for key in self.model.pars}
        fpars['dx']=pars['dx']
        mode='wrap' if self.model.periodic else 'edge'
        J=np.zeros((self.model.window_range * nvar ** 2, N))

        (idx, unknowns_idx,
         rows, cols)=get_indices(N,
                                   self.model.window_range,
                                   nvar,
                                   mode)

        unknowns=np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs=np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs=[pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ujacob in enumerate(self.ufunc):
            Ji=ujacob((*uargs.tolist() + pargs))
            J[i]=Ji
        J=sps.csr_matrix((J.T.flatten(),
                            (rows.flatten(),
                             cols.flatten())),
                           (N * self.model.nvar,
                            N * self.model.nvar))
        return J if sparse else J.todense()

    def num_approx(self, fields, pars, eps=1E-8):
        nvar, N=len(fields.vars), fields.size
        fpars={key: pars[key] for key in self.model.pars}
        fpars['dx']=pars['dx']
        J=np.zeros((N * nvar, N * nvar))
        indices=np.indices(fields.uarray.shape)
        for i, (var_index, node_index) in enumerate(zip(*map(np.ravel,
                                                             indices))):
            fields_plus=fields.copy()
            fields_plus.uarray[var_index, node_index] += eps
            fields_moins=fields.copy()
            fields_moins.uarray[var_index, node_index] -= eps
            Fplus=self.model.F(fields_plus, pars)
            Fmoins=self.model.F(fields_moins, pars)
            J[i]=(Fplus - Fmoins) / (2 * eps)

        return J.T
