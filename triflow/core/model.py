#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
import scipy.sparse as sps
from sympy import (Function, Matrix, MatrixSymbol, Symbol, Wild, symbols,
                   sympify)
from sympy.utilities.autowrap import ufuncify
from typing import Union
from toolz import memoize

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


@memoize
def get_indices(N, window_range, nvar):
    i = np.arange(N)[:, np.newaxis]
    idx = np.arange(N * nvar).reshape((nvar, N), order='F')
    idx = np.pad(idx, ((0, 0),
                       (int((window_range - 1) / 2),
                        int((window_range - 1) / 2))),
                 mode='wrap').flatten('F')
    unknowns_idx = np.arange(window_range *
                             nvar) + i * nvar
    rows = np.tile(np.arange(nvar),
                   window_range * nvar) + i * nvar
    cols = np.repeat(np.arange(window_range * nvar),
                     nvar) + i * nvar
    rows = rows
    cols = idx[cols]
    return idx, unknowns_idx, rows, cols


class Model:
    """docstring for Model"""

    def __init__(self,
                 funcs: Union[str, list, tuple, dict],
                 vars: Union[str, list, tuple],
                 pars: Union[str, list, tuple, None],
                 periodic: bool=False) -> None:
        logging.debug('enter __init__ Model')
        self.J_sparse_cache = None
        self.periodic = periodic

        self.N = Symbol('N', integer=True)
        self.x, self.dx = symbols('x dx')

        (self.funcs,
         self.vars,
         self.pars) = (funcs,
                       vars,
                       pars) = self._coerce_input(funcs, vars, pars)

        (self.symbolic_funcs,
         self.symbolic_vars,
         self.symbolic_pars) = self._sympify_model(funcs, vars, pars)

        self.total_symbolic_vars = {str(svar):
                                    {(svar, 0)} for svar in self.symbolic_vars}

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

        self.F_matrix = Matrix(F_array)
        self.J_matrix = Matrix(J_array)

    def compile(self):
        self.ufuncs_bulk = np.array([ufuncify(self.sargs,
                                              func)
                                     for func
                                     in np.array(self.F_matrix)])
        self.ujacobs_bulk = np.array([ufuncify(self.sargs,
                                               jacob)
                                      for jacob
                                      in np.array(self.J_matrix)
                                      .flatten(order='F')])

    def F(self, unknowns, pars):
        unknowns = np.array(unknowns).flatten('F')
        nvar, N = len(self.vars), int(unknowns.size / len(self.vars))
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = pars['dx']
        F = np.zeros((nvar, N))
        if self.periodic:
            unknowns = np.pad(unknowns.reshape((nvar, N), order='F'),
                              ((0, 0),
                               ((int((self.window_range - 1) / 2)),
                                int((self.window_range - 1) / 2))),
                              mode='wrap')
        uargs = np.dstack([[U[i: U.size - self.window_range + i + 1]
                            for i in range(self.window_range)]
                           for U in unknowns])
        uargs = np.rollaxis(uargs, 2, start=0)
        uargs = uargs.reshape((uargs.shape[0] * uargs.shape[1],
                               -1), order='F')

        pargs = [pars[key] for key in self.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufuncs_bulk):
            F[i, :] = ufunc((*uargs.tolist() + pargs))
        return F.flatten('F')

    def J(self, unknowns, pars):
        unknowns = np.array(unknowns).flatten('F')
        nvar, N = len(self.vars), int(unknowns.size / len(self.vars))
        fpars = {key: pars[key] for key in self.pars}
        fpars['dx'] = pars['dx']
        J = np.zeros((self.window_range * nvar ** 2, N))

        (idx, unknowns_idx,
         rows, cols) = get_indices(N,
                                   self.window_range,
                                   self.nvar)

        uargs = unknowns[idx[unknowns_idx].T]
        pargs = [pars[key] for key in self.pars] + [pars['dx']]
        for i, ujacob in enumerate(self.ujacobs_bulk):
            Ji = ujacob((*uargs.tolist() + pargs))
            J[i] = Ji
        J = sps.coo_matrix((J.T.flatten(),
                            (rows.flatten(),
                             cols.flatten())),
                           (N * self.nvar,
                            N * self.nvar))
        return J.tocsr()

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
            return (Up1 - 2 * U + Um1) / dx**2
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

    def _sympify_model(self,
                       funcs: tuple,
                       vars: tuple,
                       pars: tuple) -> tuple:
        logging.debug('enter _sympify_model')

        symbolic_vars = symbols(vars)
        symbolic_pars = symbols(pars)
        symbolic_funcs = tuple([sympify(func)
                                for func
                                in funcs])
        return symbolic_funcs, symbolic_vars, symbolic_pars

    def _approximate_derivative(self,
                                discrete_funcs: tuple,
                                discrete_vars: tuple) -> tuple:
        wild_var = Wild('var')
        wild_order = Wild('order')
        pattern = Function('dx')(wild_var, wild_order)

        logging.debug('enter _approximate_derivative')
        approximated_funcs = []
        for dfunc in discrete_funcs:
            afunc = dfunc
            derivatives = dfunc.find(pattern)
            for derivative in derivatives:
                matched = derivative.match(pattern)
                var = matched[wild_var]
                order = matched[wild_order]
                afunc = afunc.replace(
                    derivative,
                    self.finite_diff_scheme(var,
                                            order))
            approximated_funcs.append(afunc)
        return tuple(approximated_funcs)
