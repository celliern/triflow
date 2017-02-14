#!/usr/bin/env python
# coding=utf8

import logging
from itertools import product

import numpy as np
import scipy.sparse as sps
from sympy import (Derivative, Function, Matrix, Symbol, symbols,
                   sympify)
from sympy.utilities.autowrap import ufuncify
from toolz import memoize
from typing import Union
from recordclass import recordclass

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def generate_fields_container(vars, fields):
    rc = recordclass('Fields', ['x'] + list(vars) + list(fields))

    class Fields(rc):

        def __init__(self, *args, **kwargs):
            self.vars = vars
            self.fields = fields
            self.size = len(self.x)
            self.keys = ['x'] + list(vars) + list(fields)
            data = list(zip(*[getattr(self, var)
                              for var in self.keys]))
            self.array = np.array(data)
            self.dtype = [(var, float) for var in self.keys]
            for var in self.keys:
                self.__setattr__(var, self.rec[var])

        @property
        def flat(self):
            return self.array.ravel()

        @property
        def rec(self):
            return self.array.view(dtype=self.dtype)

        @property
        def uarray(self):
            return self.array[:, 1: (1 + len(self.vars))]

        @property
        def uflat(self):
            uflat = self.array[:, 1: (1 + len(self.vars))].ravel()
            uflat.flags.writeable = False
            return uflat

        def fill(self, Uflat):
            new_fields = self.copy()
            new_fields.uarray[:] = Uflat.reshape(new_fields.uarray.shape)
            return new_fields

        def __getitem__(self, index):
            return self.rec[index]

        def __iter__(self):
            return (self.array[i] for i in range(self.size))

        def copy(self):
            NewField = generate_fields_container(self.vars,
                                                 self.fields)
            old_values = {var: getattr(self, var).squeeze()
                          for var in self.keys}
            return NewField(**old_values)

        def __repr__(self):
            return self.rec.__repr__()

    return Fields


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


class H_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        N = fields.size
        middle_point = int((self.model.window_range - 1) / 2)
        fpars = {key: pars[key] for key in self.model.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if self.model.periodic else 'edge'
        F = np.zeros((1, N))
        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs = [pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufunc):
            F[i] = ufunc((*uargs.tolist() + pargs)).squeeze()
        return F.flatten('F')


class F_Routine(ModelRoutine):
    def __call__(self, fields, pars):
        nvar, N = len(fields.vars), fields.size
        middle_point = int((self.model.window_range - 1) / 2)
        fpars = {key: pars[key] for key in self.model.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if self.model.periodic else 'edge'
        F = np.zeros((nvar, N))
        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs = [pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ufunc in enumerate(self.ufunc):
            F[i, :] = ufunc((*uargs.tolist() + pargs))
        return F.flatten('F')


class J_Routine(ModelRoutine):
    def __call__(self, fields, pars, sparse=True):
        nvar, N = len(fields.vars), fields.size
        middle_point = int((self.model.window_range - 1) / 2)
        fpars = {key: pars[key] for key in self.model.pars}
        fpars['dx'] = pars['dx']
        mode = 'wrap' if self.model.periodic else 'edge'
        J = np.zeros((self.model.window_range * nvar ** 2, N))

        (idx, unknowns_idx,
         rows, cols) = get_indices(N,
                                   self.model.window_range,
                                   nvar,
                                   mode)

        unknowns = np.pad(fields.array,
                          ((middle_point, middle_point),
                           (0, 0)), mode=mode)
        uargs = np.concatenate([fields.x] +
                               [unknowns[i: N + i, 1:]
                                for i in range(self.model.window_range)],
                               axis=1).T
        pargs = [pars[key] for key in self.model.pars] + [pars['dx']]
        for i, ujacob in enumerate(self.ufunc):
            Ji = ujacob((*uargs.tolist() + pargs))
            J[i] = Ji
        J = sps.csr_matrix((J.T.flatten(),
                            (rows.flatten(),
                             cols.flatten())),
                           (N * self.model.nvar,
                            N * self.model.nvar))
        return J if sparse else J.todense()

    def num_approx(self, fields, pars, eps=1E-8):
        nvar, N = len(fields.vars), fields.size
        fpars = {key: pars[key] for key in self.model.pars}
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


class Model:
    """docstring for Model"""

    def __init__(self,
                 funcs: Union[str, list, tuple, dict],
                 vars: Union[str, list, tuple],
                 pars: Union[str, list, tuple, None]=None,
                 fields: Union[str, list, tuple, None]=None,
                 helpers: Union[dict, tuple, None]=None,
                 periodic: bool=False) -> None:
        self.N = Symbol('N', integer=True)
        x, dx = self.x, self.dx = symbols('x dx')
        y, dy = self.y, self.dy = symbols('y dy')

        logging.debug('enter __init__ Model')
        self.periodic = periodic

        (self.funcs,
         self.vars,
         self.pars,
         self.fields,
         self.helpers) = (funcs,
                          vars,
                          pars,
                          fields,
                          helpers) = self._coerce_inputs(funcs, vars,
                                                         pars, fields,
                                                         helpers)

        self.sympify_namespace = {}
        self.sympify_namespace.update(self.generate_sympify_namespace(
            'x',
            self.vars,
            self.fields))

        logging.debug(self.sympify_namespace)
        (self.symbolic_funcs,
         self.symbolic_vars,
         self.symbolic_pars,
         self.symbolic_fields,
         self.symbolic_helpers) = self._sympify_model(funcs, vars,
                                                      pars, fields, helpers)

        self.total_symbolic_vars = {str(svar.func):
                                    {(svar.func, 0)}
                                    for svar in (self.symbolic_vars +
                                                 self.symbolic_fields)}

        approximated_funcs = self._approximate_derivative(self.symbolic_funcs,
                                                          self.symbolic_vars,
                                                          self.symbolic_fields)
        approximated_helpers = {key: self._approximate_derivative(
            [value],
            self.symbolic_vars,
            self.symbolic_fields
        ) for key, value in self.symbolic_helpers.items()}
        self.bounds = bounds = self._extract_bounds(vars,
                                                    self.total_symbolic_vars)
        self.window_range = bounds[-1] - bounds[0] + 1
        self.nvar = len(vars)
        self.unknowns = unknowns = self._extract_unknowns(
            vars,
            bounds, self.total_symbolic_vars).flatten('F')

        F_array = np.array(approximated_funcs)
        J_array = np.array([
            [func.diff(unknown)
                for unknown in unknowns]
            for func in approximated_funcs])

        self.dfields = self._extract_unknowns(
            vars + fields,
            bounds, self.total_symbolic_vars).flatten('F')

        self.F = F_Routine(Matrix(F_array), self)
        self.H = {key: H_Routine(value, self)
                  for key, value in approximated_helpers.items()}
        self.J = J_Routine(Matrix(J_array), self)
        self.Fields = generate_fields_container(vars, fields, helpers)

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
        return ([self.x] +
                list(self.dfields) +
                list(self.symbolic_pars) + [self.dx])

    def generate_sympify_namespace(self, independent_variable, vars, fields):
        symbolic_independent_variable = Symbol(independent_variable)
        namespace = {independent_variable: symbolic_independent_variable}
        namespace.update({'d%s' % (independent_variable * i):
                          lambda U: Derivative(U,
                                               symbolic_independent_variable,
                                               i)
                          for i in range(1, 5)})
        namespace.update({'d%s%s' % (independent_variable * order, var):
                          Derivative(Function(var)(independent_variable),
                                     independent_variable, order)
                          for order, var in product(range(1, 5),
                                                    vars + fields)})
        return namespace

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

    def _coerce_inputs(self, funcs, vars, pars, fields, helpers):
        pars = tuple(pars) if pars is not None else tuple()
        fields = tuple(fields) if fields is not None else tuple()
        helpers = helpers if helpers is not None else {}

        if isinstance(funcs, (str, )):
            funcs = [funcs]
        if isinstance(vars, (str, )):
            vars = [vars]
        if isinstance(fields, (str, )):
            fields = [fields]
        if isinstance(pars, (str, )):
            pars = [pars]
        if isinstance(funcs, (dict, )):
            funcs = [funcs[key] for key in vars if key in funcs.keys()]

        funcs = tuple(funcs)
        vars = tuple(vars)
        fields = tuple(fields)
        return funcs, vars, pars, fields, helpers

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
                       pars: tuple,
                       fields: tuple,
                       helpers: dict) -> tuple:
        logging.debug('enter _sympify_model')

        symbolic_vars = tuple([Function(var)(self.x) for var in vars])
        symbolic_fields = tuple([Function(field)(self.x) for field in fields])
        symbolic_pars = symbols(pars)
        symbolic_helpers = {key:
                            sympify(value, locals=self.sympify_namespace)
                            .subs(zip(map(Symbol, vars),
                                      (symbolic_vars +
                                       symbolic_fields)))
                            for key, value
                            in helpers.items()}
        symbolic_funcs = tuple([sympify(func,
                                        locals=self.sympify_namespace)
                                .subs({Symbol(key): value
                                       for key, value
                                       in symbolic_helpers.items()})
                                .subs(zip(map(Symbol, vars),
                                          (symbolic_vars +
                                           symbolic_fields)))
                                for func
                                in funcs])
        return (symbolic_funcs, symbolic_vars,
                symbolic_pars, symbolic_fields, symbolic_helpers)

    def _approximate_derivative(self,
                                symbolic_funcs: tuple,
                                symbolic_vars: tuple,
                                symbolic_fields: tuple) -> tuple:

        logging.debug('enter _approximate_derivative')
        approximated_funcs = []
        for func in symbolic_funcs:
            afunc = func
            for derivative in func.find(Derivative):
                var = Symbol(str(derivative.args[0].func))
                logging.debug(f"{derivative}, {var}")
                order = len(derivative.args) - 1
                afunc = afunc.replace(
                    derivative,
                    self.finite_diff_scheme(var,
                                            order))
            afunc = afunc.subs([(var, Symbol(str(var.func)))
                                for var in symbolic_vars + symbolic_fields])
            approximated_funcs.append(afunc)
        return tuple(approximated_funcs)
