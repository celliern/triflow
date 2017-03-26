#!/usr/bin/env python
# coding=utf8

import logging
import sys
from functools import partial
from itertools import product
from pickle import dump, load

import numpy as np
import theano as th
import theano.sparse as ths
from sympy import Derivative, Function, Symbol, symbols, sympify
from sympy.printing.theanocode import theano_code
from theano import tensor as T
from theano.ifelse import ifelse
from triflow.core.routines import F_Routine, J_Routine
from typing import Union

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

sys.setrecursionlimit(40000)


def partial_derivative(symbolic_independent_variable, i, U):
    return Derivative(U, symbolic_independent_variable, i)


def generate_sympify_namespace(independent_variable, vars, fields):
    symbolic_independent_variable = Symbol(independent_variable)
    namespace = {independent_variable: symbolic_independent_variable}
    namespace.update({'d%s' % (independent_variable * i):
                      partial(symbolic_independent_variable, i)
                      for i in range(1, 5)})
    namespace.update({'d%s%s' % (independent_variable * order, var):
                      Derivative(Function(var)(independent_variable),
                                 independent_variable, order)
                      for order, var in product(range(1, 5),
                                                vars + fields)})
    return namespace


def generate_fields_container(vars, fields):

    class Fields:
        def __init__(self, **kwargs):
            assert set(kwargs.keys()) == set(['x'] +
                                             list(vars) +
                                             list(fields))
            for key, value in kwargs.items():
                self.__setattr__(key, value)
            self.reduce_container = kwargs
            self.vars = vars
            self.fields = fields
            self.size = len(self.x)
            self.keys = ['x'] + list(vars) + list(fields)
            data = list(zip(*[getattr(self, var)
                              for var in self.keys]))
            self.array = np.array(data)
            self.dtype = [(var, float) for var in self.keys]
            for var in self.keys:
                self.__setattr__(var, self.rec[var].squeeze())

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
            self.uarray[:] = Uflat.reshape(self.uarray.shape)

        def __getitem__(self, index):
            return self.rec[index].squeeze()

        def __iter__(self):
            return (self.array[i] for i in range(self.size))

        def copy(self):
            old_values = {var: getattr(self, var).squeeze()
                          for var in self.keys}
            # NewField = generate_fields_container(self.vars, self.fields)

            return self.__class__(**old_values)

        def __repr__(self):
            return self.rec.__repr__()

    return Fields


def load_model(filename):
    with open(filename, 'rb') as f:
        return load(f)


def reduce_model(funcs, vars, pars,
                 fields, helpers,
                 f, j):
    model = Model(funcs, vars, pars,
                  fields, helpers, reduced=True)
    model.th_routines = [f, j]
    model._compile(model.F_array, model.J_array, f, j)

    return model


class Model:
    """docstring for Model"""

    def __init__(self,
                 funcs: Union[str, list, tuple, dict],
                 vars: Union[str, list, tuple],
                 pars: Union[str, list, tuple, None]=None,
                 fields: Union[str, list, tuple, None]=None,
                 helpers: Union[dict, tuple, None]=None,
                 reduced=False) -> None:
        self.N = Symbol('N', integer=True)
        x, dx = self.x, self.dx = symbols('x dx')
        y, dy = self.y, self.dy = symbols('y dy')

        logging.debug('enter __init__ Model')

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

        sympify_namespace = {}
        sympify_namespace.update(generate_sympify_namespace(
            'x',
            self.vars,
            self.fields))

        (self.symbolic_funcs,
         self.symbolic_vars,
         self.symbolic_pars,
         self.symbolic_fields,
         self.symbolic_helpers) = self._sympify_model(funcs, vars,
                                                      pars, fields, helpers,
                                                      sympify_namespace)

        self.total_symbolic_vars = {str(svar.func):
                                    {(svar.func, 0)}
                                    for svar in (self.symbolic_vars +
                                                 self.symbolic_fields)}

        approximated_funcs = self._approximate_derivative(self.symbolic_funcs,
                                                          self.symbolic_vars,
                                                          self.symbolic_fields)
        self.bounds = bounds = self._extract_bounds(vars,
                                                    self.total_symbolic_vars)
        self.window_range = bounds[-1] - bounds[0] + 1
        self.nvar = len(vars)
        self.unknowns = unknowns = self._extract_unknowns(
            vars,
            bounds, self.total_symbolic_vars).flatten('F')

        self.F_array = np.array(approximated_funcs)
        self.J_array = np.array([
            [func.diff(unknown).expand()
                for unknown in unknowns]
            for func in approximated_funcs]).flatten('F')
        self.sparse_indices = np.where(self.J_array != 0)

        self.J_array = self.J_array[self.sparse_indices]

        self.dfields = self._extract_unknowns(
            vars + fields,
            bounds, self.total_symbolic_vars).flatten('F')

        if not reduced:
            F, J, args, self.map_extended = self._theano_convert(self.F_array,
                                                                 self.J_array)
            f = th.function(inputs=args,
                            outputs=F,
                            on_unused_input='ignore')
            j = th.function(inputs=args,
                            outputs=J,
                            on_unused_input='ignore')
            self.th_args = args
            self.th_vectors = [F, J]
            self.th_routines = [f, j]
            self._compile(self.F_array, self.J_array, f, j)

    def _theano_convert(self, F_array, J_array):
        th_args = list(map(partial(theano_code,
                                   broadcastables={arg: (False,)
                                                   for arg
                                                   in [self.x,
                                                       *self.dfields,
                                                       *self.symbolic_pars]}),
                           self.sargs))
        ins = th.gof.graph.inputs(th_args)
        mapargs = {inp.name: inp
                   for inp in ins if isinstance(inp, T.TensorVariable)}
        x_th = T.dvector('x')
        periodic = T.bscalar('periodic')
        middle_point = int((self.window_range - 1) / 2)
        nvar = len(self.vars)
        N = x_th.size
        L = x_th[-1]
        computed_dx = L / N
        bounds = self.bounds
        window_range = self.window_range
        subs = {mapargs['dx']: computed_dx}
        map_extended = {}
        for varname, discretisation_tree in self.total_symbolic_vars.items():
            pad_left, pad_right = bounds
            per_extended_var = T.concatenate([mapargs[varname][pad_left:],
                                              mapargs[varname],
                                              mapargs[varname][:pad_right]])
            edge_extended_var = T.concatenate([T.repeat(mapargs[varname][:1],
                                                        middle_point),
                                               mapargs[varname],
                                               T.repeat(mapargs[varname][-1:],
                                                        middle_point)])
            extended_var = ifelse(periodic,
                                  per_extended_var,
                                  edge_extended_var)
            map_extended[varname] = extended_var
            for order in range(pad_left, pad_right + 1):
                if order != 0:
                    var = (f"{varname}_{'m' if order < 0 else 'p'}"
                           f"{np.abs(order)}")
                    new_var = extended_var[order - pad_left:
                                           extended_var.size +
                                           order - pad_right]
                    subs.update({mapargs[var]:
                                 new_var})
        F = list(map(partial(theano_code,
                             broadcastables={arg: (False,)
                                             for arg
                                             in [self.x,
                                                 *self.dfields,
                                                 *self.symbolic_pars]}),
                     F_array.flatten().tolist()))

        J = list(map(partial(theano_code,
                             broadcastables={arg: (False,)
                                             for arg
                                             in [self.x,
                                                 *self.dfields,
                                                 *self.symbolic_pars]}),
                     J_array.flatten().tolist()))
        F = (T.concatenate(th.clone(F, replace=subs))
             .reshape((nvar, N)).T.flatten())
        J = th.clone(J, replace=subs)

        sparse_indices = self.sparse_indices

        J = T.stack([T.repeat(j, N) if j.ndim == 0 else j for j in J]).T
        i = T.arange(N).dimshuffle([0, 'x'])
        idx = T.arange(N * nvar).reshape((N, nvar)).T
        edge_extended_idx = T.concatenate([T.repeat(idx[:, :1],
                                                    middle_point,
                                                    axis=1),
                                           idx,
                                           T.repeat(idx[:, -1:],
                                                    middle_point,
                                                    axis=1)],
                                          axis=1).T.flatten()
        per_extended_idx = T.concatenate([idx[:, -middle_point:],
                                          idx,
                                          idx[:, :middle_point]],
                                         axis=1).T.flatten()
        extended_idx = ifelse(periodic,
                              per_extended_idx, edge_extended_idx)

        rows = T.tile(T.arange(nvar),
                      window_range * nvar) + i * nvar
        cols = T.repeat(T.arange(window_range * nvar),
                        nvar) + i * nvar
        rows = rows[:, sparse_indices].reshape(J.shape).flatten()
        cols = extended_idx[cols][:, sparse_indices].reshape(J.shape).flatten()

        permutation = T.argsort(cols)

        J = J.flatten()[permutation]
        rows = rows[permutation]
        cols = cols[permutation]
        count = T.zeros((N * nvar + 1,), dtype=int)
        uq, cnt = T.extra_ops.Unique(False, False, True)(cols)
        count = T.set_subtensor(count[uq + 1], cnt)

        indptr = T.cumsum(count)
        shape = T.stack([N * nvar, N * nvar])
        sparse_J = ths.CSC(J, rows, indptr, shape)

        th_args = [x_th, *[mapargs[key]
                           for key
                           in self.vars + self.fields + self.pars],
                   periodic]

        return F, sparse_J, th_args, map_extended

    def _compile(self, F_array, J_array, f, j):
        logging.debug('compile F')
        self.F = F_Routine(F_array, self.vars + self.fields,
                           self.pars, f)
        logging.debug('compile J')
        self.J = J_Routine(J_array, self.vars + self.fields,
                           self.pars, j)

    @property
    def fields_template(self):
        return generate_fields_container(self.vars, self.fields)

    @property
    def args(self):
        return map(str, self.sargs)

    @property
    def sargs(self):
        return ([self.x] +
                list(self.dfields) +
                list(self.symbolic_pars) + [self.dx])

    def save(self, filename):
        with open(filename, 'wb') as f:
            return dump(self, f)

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
                       helpers: dict,
                       sympify_namespace: dict) -> tuple:
        logging.debug('enter _sympify_model')

        symbolic_vars = tuple([Function(var)(self.x) for var in vars])
        symbolic_fields = tuple([Function(field)(self.x) for field in fields])
        symbolic_pars = symbols(pars)
        symbolic_helpers = {key:
                            sympify(value, locals=sympify_namespace)
                            .subs(zip(map(Symbol, vars),
                                      (symbolic_vars +
                                       symbolic_fields)))
                            for key, value
                            in helpers.items()}
        symbolic_funcs = tuple([sympify(func, locals=sympify_namespace)
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
            approximated_funcs.append(afunc.expand())
        return tuple(approximated_funcs)

    def __reduce__(self):
        f, j = self.th_routines
        return (reduce_model, (self.funcs, self.vars,
                               self.pars, self.fields,
                               self.helpers, f, j
                               ))
