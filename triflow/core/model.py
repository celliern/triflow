#!/usr/bin/env python
# coding=utf8

import logging

from sympy import (Idx, IndexedBase, Symbol, Wild, symbols,
                   sympify, Function, Eq, Matrix, MatrixSymbol)
from sympy.utilities.codegen import codegen
from typing import Union

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Model:
    """docstring for Model"""

    def __init__(self,
                 funcs: Union[str, list, tuple, dict],
                 vars: Union[str, list, tuple],
                 pars: Union[str, list, tuple, None]) -> None:
        logging.debug('enter __init__ Model')

        self.N = Symbol('N', integer=True)
        self.i = Idx('i', (0, self.N))
        self.x, self.dx = symbols('x dx')
        self.discrete_x = IndexedBase('x')[self.i]
        self.total_symbolic_vars = set()

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
        (symbolic_funcs,
         symbolic_vars,
         symbolic_pars) = self._sympify_model(funcs, vars, pars)
        self.total_symbolic_vars.update({(svar, 0) for svar in symbolic_vars})
        approximated_funcs = self._approximate_derivative(symbolic_funcs,
                                                          symbolic_vars)

        self.total_symbolic_vars = [tdvar[0]
                                    for tdvar in sorted(
            list(self.total_symbolic_vars),
            key=lambda x: [x[1],
                           str(x[0])])]

        F_vector = Matrix(approximated_funcs)
        J_matrix = Matrix([[func.diff(var) for var in self.total_symbolic_vars]
                           for func in approximated_funcs])
        print(F_vector.shape)
        print(J_matrix.shape)

        discrete_vars = self._discretize_vars(self.total_symbolic_vars)
        F_vector = F_vector.subs({svar: dvar
                                  for svar, dvar in zip(symbolic_vars,
                                                        discrete_vars)})
        J_matrix = J_matrix.subs({svar: dvar
                                  for svar, dvar in zip(symbolic_vars,
                                                        discrete_vars)})

        print(Eq(MatrixSymbol('F', F_vector.shape), F_vector))

    def finite_diff_scheme(self, U, order):
        dx = Symbol('dx')
        var_label = str(U)
        if order == 1:
            Um1 = Symbol('%s_m1' % var_label)
            Up1 = Symbol('%s_p1' % var_label)
            self.total_symbolic_vars.add((Um1, -1))
            self.total_symbolic_vars.add((Up1, 1))
            return (1 / 2 * Up1 - 1 / 2 * Um1) / dx
        if order == 2:
            Um1 = Symbol('%s_m1' % var_label)
            Up1 = Symbol('%s_p1' % var_label)
            self.total_symbolic_vars.add((Um1, -1))
            self.total_symbolic_vars.add((Up1, 1))
            return (Up1 - 2 * U + Um1) / dx**2
        if order == 3:
            Um1, Um2 = Symbol('%s_m:2' % var_label)
            Up1, Up2 = Symbol('%s_p:2}' % var_label)
            self.total_symbolic_vars.add((Um1, -1))
            self.total_symbolic_vars.add((Up1, 1))
            self.total_symbolic_vars.add((Um2, -2))
            self.total_symbolic_vars.add((Up2, 2))
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

    def _discretize_vars(self,
                         symbolic_vars: tuple) -> tuple:
        logging.debug('enter _discretize_vars')
        discrete_vars = tuple([IndexedBase(svar)[self.i]
                               for svar
                               in symbolic_vars])
        return discrete_vars

    def _approximate_derivative(self,
                                discrete_funcs: tuple,
                                discrete_vars: tuple) -> tuple:
        wild_var = Wild('var')
        wild_order = Wild('order')
        pattern = Function('dx')(wild_var, wild_order)

        logging.debug('enter _discretize_func')
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
