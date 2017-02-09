#!/usr/bin/env python
# coding=utf8

import logging
from itertools import product

import numpy as np
import scipy.sparse as sps
from sympy import (Derivative, Function, Matrix, MatrixSymbol, Symbol, symbols,
                   sympify)
from sympy.utilities.autowrap import ufuncify
from toolz import memoize
from typing import Union

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


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


class ModelRoutine:
    def __init__(self, matrix, model):
        self.matrix = matrix
        self.args = model.sargs
        self.ufunc = np.array([ufuncify(self.args, func)
                               for func
                               in np.array(self.matrix)
                               .flatten(order='F')])
        self.model = model

    def __repr__(self):
        return self.matrix.__repr__()


class F_Routine(ModelRoutine):
    def __call__(self, unknowns, pars):
        unknowns = np.array(unknowns).flatten('F')
        nvar, N = len(self.model.vars), int(unknowns.size /
                                            len(self.model.vars))
        fpars = {key: pars[key] for key in self.model.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if self.model.periodic else 'edge'
        F = np.zeros((nvar, N))

        unknowns = np.pad(unknowns.reshape((nvar, N), order='F'),
                          ((0, 0),
                           ((int((self.model.window_range - 1) / 2)),
                            int((self.model.window_range - 1) / 2))),
                          mode=mode)
        uargs = np.dstack([[U[i: U.size - self.model.window_range + i + 1]
                            for i in range(self.model.window_range)]
                           for U in unknowns])
        uargs = np.rollaxis(uargs, 2, start=0)
        uargs = uargs.reshape((uargs.shape[0] * uargs.shape[1],
                               -1), order='F')

        pargs = [pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufunc):
            F[i, :] = ufunc((*uargs.tolist() + pargs))
        return F.flatten('F')


class J_Routine(ModelRoutine):
    def __call__(self, unknowns, pars, sparse=True):
        unknowns = np.array(unknowns).flatten('F')
        nvar, N = len(self.model.vars), int(unknowns.size /
                                            len(self.model.vars))
        fpars = {key: pars[key] for key in self.model.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if self.model.periodic else 'edge'
        J = np.zeros((self.model.window_range * nvar ** 2, N))

        (idx, unknowns_idx,
         rows, cols) = get_indices(N,
                                   self.model.window_range,
                                   self.model.nvar,
                                   mode)

        uargs = unknowns[idx[unknowns_idx].T]
        pargs = [pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ujacob in enumerate(self.ufunc):
            Ji = ujacob((*uargs.tolist() + pargs))
            J[i] = Ji
        J = sps.coo_matrix((J.T.flatten(),
                            (rows.flatten(),
                             cols.flatten())),
                           (N * self.model.nvar,
                            N * self.model.nvar))
        return J.tocsr() if sparse else J.todense()


class Model:
    """docstring for Model"""

    def __init__(self,
                 funcs: Union[str, list, tuple, dict],
                 vars: Union[str, list, tuple],
                 pars: Union[str, list, tuple, None],
                 periodic: bool=False) -> None:
        self.N = Symbol('N', integer=True)
        x, dx = self.x, self.dx = symbols('x dx')
        self.sympify_namespace = {'x': self.x,
                                  'dx': lambda U: Derivative(U, x),
                                  'dxx': lambda U: Derivative(U, x, x),
                                  'dxxx': lambda U: Derivative(U, x, x, x),
                                  'dxxxx': lambda U: Derivative(U, x, x, x, x)}
        self.sympify_namespace.update({'d%s%s' % ('x' * order, var):
                                       Derivative(Symbol(var), x, order)
                                       for order, var
                                       in product(range(1, 5), vars)})
        logging.debug('enter __init__ Model')
        self.J_sparse_cache = None
        self.periodic = periodic

        (self.funcs,
         self.vars,
         self.pars) = (funcs,
                       vars,
                       pars) = self._coerce_input(funcs, vars, pars)

        (self.symbolic_funcs,
         self.symbolic_vars,
         self.symbolic_pars) = self._sympify_model(funcs, vars, pars)

        self.total_symbolic_vars = {str(svar.func):
                                    {(svar.func, 0)}
                                    for svar in self.symbolic_vars}

        approximated_funcs = self._approximate_derivative(self.symbolic_funcs,
                                                          self.symbolic_vars)

        self.bounds = bounds = self._extract_bounds(vars,
                                                    self.total_symbolic_vars)
        self.window_range = bounds[-1] - bounds[0] + 1
        self.nvar = len(vars)
        self.unknowns = unknowns = self._extract_unknowns(
            vars, bounds, self.total_symbolic_vars).flatten(order='F')

        F_array = np.array(approximated_funcs)
        J_array = np.array([
            [func.diff(unknown)
                for unknown in unknowns]
            for func in approximated_funcs])

        self.F = F_Routine(Matrix(F_array), self)
        self.J = J_Routine(Matrix(J_array), self)

    def set_periodic_bdc(self):
        self.periodic = True

    def set_left_bdc(self, kind, value):
        pass

    def set_right_bdc(self, kind, value):
        pass

    @property
    def args(self):
        return map(str, self.sargs)

    @property
    def sargs(self):
        return list(self.unknowns) + list(self.symbolic_pars) + [self.dx]

    @property
    def dargs(self):
        return map(lambda arg: MatrixSymbol(arg, self.N, 1), self.sargs)

    def _extract_bounds(self, vars, dict_symbol):
        bounds = (0, 0)
        for var in vars:
            dvars, orders = zip(*dict_symbol[var])

            bounds = (min(bounds[0],
                          min(orders)),
                      max(bounds[1],
                          max(orders))
                      )
        return bounds

    def _extract_unknowns(self, vars, bounds, dict_symbol):
        unknowns = np.zeros((len(vars), bounds[-1] - bounds[0] + 1),
                            dtype=object)
        for i, var in enumerate(vars):
            dvars, orders = zip(*dict_symbol[var])
            for j, order in enumerate(range(bounds[0], bounds[1] + 1)):
                if order == 0:
                    unknowns[i, j] = Symbol(var)
                if order < 0:
                    unknowns[i, j] = Symbol(f'{var}_m{np.abs(order)}')
                if order > 0:
                    unknowns[i, j] = Symbol(f'{var}_p{np.abs(order)}')
        return unknowns

    def _coerce_input(self, funcs, vars, pars):

        if isinstance(funcs, (str, )):
            funcs = [funcs]
        if isinstance(vars, (str, )):
            vars = [vars]
        if isinstance(pars, (str, )):
            pars = [pars]
        if isinstance(funcs, (dict, )):
            funcs = [funcs[key] for key in vars if key in funcs.keys()]

        funcs = tuple(funcs)
        vars = tuple(vars)
        pars = tuple(pars) if pars is not None else tuple()
        return funcs, vars, pars

    def finite_diff_scheme(self, U, order):
        dx = Symbol('dx')
        var_label = str(U)
        if order == 1:
            Um1 = Symbol('%s_m1' % var_label)
            Up1 = Symbol('%s_p1' % var_label)
            self.total_symbolic_vars[var_label].add((Um1, -1))
            self.total_symbolic_vars[var_label].add((Up1, 1))
            return (1 / 2 * Up1 - 1 / 2 * Um1) / dx
        if order == 2:
            Um1 = Symbol('%s_m1' % var_label)
            Up1 = Symbol('%s_p1' % var_label)
            self.total_symbolic_vars[var_label].add((Um1, -1))
            self.total_symbolic_vars[var_label].add((Up1, 1))
            return (Up1 - 2 * U + Um1) / dx ** 2
        if order == 3:
            Um1 = Symbol('%s_m1' % var_label)
            Um2 = Symbol('%s_m2' % var_label)
            Up1 = Symbol('%s_p1' % var_label)
            Up2 = Symbol('%s_p2' % var_label)
            self.total_symbolic_vars[var_label].add((Um1, -1))
            self.total_symbolic_vars[var_label].add((Up1, 1))
            self.total_symbolic_vars[var_label].add((Um2, -2))
            self.total_symbolic_vars[var_label].add((Up2, 2))
            return (-1 / 2 * Um2 + Um1 - Up1 + 1 / 2 * Up2) / dx ** 3
        if order == 4:
            Um1 = Symbol('%s_m1' % var_label)
            Um2 = Symbol('%s_m2' % var_label)
            Up1 = Symbol('%s_p1' % var_label)
            Up2 = Symbol('%s_p2' % var_label)
            self.total_symbolic_vars[var_label].add((Um1, -1))
            self.total_symbolic_vars[var_label].add((Up1, 1))
            self.total_symbolic_vars[var_label].add((Um2, -2))
            self.total_symbolic_vars[var_label].add((Up2, 2))
            return (Um2 - 4 * Um1 + 6 * U - 4 * Up1 + Up2) / dx ** 4
        raise NotImplementedError('Finite difference up'
                                  'to 5th order not implemented yet')

    def _sympify_model(self,
                       funcs: tuple,
                       vars: tuple,
                       pars: tuple) -> tuple:
        logging.debug('enter _sympify_model')

        symbolic_vars = tuple([Function(var)(self.x) for var in vars])
        symbolic_pars = symbols(pars)
        symbolic_funcs = tuple([sympify(func,
                                        locals=self.sympify_namespace)
                                .subs(zip(map(Symbol, vars),
                                          symbolic_vars)).doit()
                                for func
                                in funcs])
        return symbolic_funcs, symbolic_vars, symbolic_pars

    def _approximate_derivative(self,
                                symbolic_funcs: tuple,
                                symbolic_vars: tuple) -> tuple:

        logging.debug('enter _approximate_derivative')
        approximated_funcs = []
        for func in symbolic_funcs:
            afunc = func
            for derivative in func.find(Derivative):
                var = Symbol(str(derivative.args[0].func))
                order = len(derivative.args) - 1
                afunc = afunc.replace(
                    derivative,
                    self.finite_diff_scheme(var,
                                            order))
            afunc = afunc.subs([(var, Symbol(str(var.func)))
                                for var in symbolic_vars])
            approximated_funcs.append(afunc)
        return tuple(approximated_funcs)
