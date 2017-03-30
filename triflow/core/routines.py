#!/usr/bin/env python
# coding=utf8

import numpy as np
import sympy as sp


class ModelRoutine:
    def __init__(self, matrix, args, pars, ufunc,
                 reduced=False):
        self.pars = list(pars) + ['periodic']
        self.matrix = matrix
        self.args = args
        self.ufunc = ufunc

    def __repr__(self):
        return sp.Matrix(self.matrix.tolist()).__repr__()


class F_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        uargs = [fields.x, *[fields[key] for key in self.args]]
        pargs = [pars[key] + fields.x * 0 if key != 'periodic' else pars[key]
                 for key
                 in self.pars]
        F = self.ufunc(*uargs, *pargs)
        return F

    def diff_approx(self, fields, pars, eps=1E-8):
        nvar, N = len(fields.vars), fields.size
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = (fields.x[-1] - fields.x[0]) / fields.x.size
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
    def __call__(self, fields, pars, sparse=True):
        uargs = [fields.x, *[fields[key] for key in self.args]]
        pargs = [pars[key] + fields.x * 0 if key != 'periodic' else pars[key]
                 for key
                 in self.pars]
        J = self.ufunc(*uargs, *pargs)

        return J if sparse else J.todense()
