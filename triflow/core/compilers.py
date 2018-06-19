#!/usr/bin/env python
# coding=utf8

from itertools import chain

import theano.tensor as tt
from sympy import And, Idx, Integer, Symbol
from sympy.printing.theanocode import TheanoPrinter, dim_handling, mapping

from .system import DependentVariable, IndependentVariable


def th_depvar_printer(printer, dvar):
    kwargs = dict(dtypes=printer._dtypes, broadcastables=printer._broadcast)
    return printer.doprint(dvar.discrete, **kwargs)


def th_ivar_printer(printer, ivar):
    kwargs = dict(dtypes=printer._dtypes, broadcastables=printer._broadcast)
    return (printer.doprint(ivar.discrete, **kwargs),
            printer.doprint(Symbol(str(ivar.idx)), **kwargs),
            printer.doprint(ivar.step, **kwargs),
            printer.doprint(ivar.N, **kwargs))


def idx_to_symbol(idx, range):
    return Symbol(str(idx))


def theano_and(*args):
    return tt.all(args, axis=0)


mapping[And] = theano_and


class EnhancedTheanoPrinter(TheanoPrinter):
    def _init_broadcast(self, system):
        dvar_broadcast = {dvar.discrete:
                          list(dim_handling([dvar.discrete],
                                            dim=len(dvar.independent_variables)
                                            ).values())[0]
                          for dvar in system.dependent_variables}
        ivar_broadcast = dim_handling(
            list(chain(*[(ivar.idx, ivar.discrete)
                         for ivar in system.independent_variables])),
            dim=1)
        return {Symbol(str(key)): value
                for key, value
                in chain(dvar_broadcast.items(),
                         ivar_broadcast.items())}

    def _init_dtypes(self, system):
        sizes_dtypes = {
            ivar.N: "int16" for ivar in system.independent_variables}
        idx_dtypes = {Symbol(str(ivar.idx)): "int16"
                      for ivar in system.independent_variables}
        return {key: value for key, value in chain(sizes_dtypes.items(),
                                                   idx_dtypes.items())}

    def __init__(self, system, *args, **kwargs):
        self._system = system
        self._broadcast = self._init_broadcast(system)
        self._dtypes = self._init_dtypes(system)
        super().__init__(*args, **kwargs)

    def _print_Idx(self, idx, **kwargs):
        try:
            return tt.constant([self._print_Integer(Integer(str(idx)))],
                               dtype="int16", ndim=1)
        except ValueError:
            pass
        idx_symbol = idx.replace(Idx, idx_to_symbol)
        idx_th = self._print(idx_symbol, dtypes=self._dtypes,
                             broadcastables=self._broadcast)
        return idx_th

    def _print_IndexedBase(self, indexed_base, ndim=1, **kwargs):
        base_symbol = Symbol(str(indexed_base))
        return self._print_Symbol(base_symbol, dtypes=self._dtypes,
                                  broadcastables=self._broadcast)

    def _print_Indexed(self, indexed, **kwargs):
        idxs = [self._print(idx,
                            dtypes=self._dtypes,
                            broadcastables=self._broadcast)
                for idx in indexed.indices]
        idxs = [tt.cast(idx, dtype="int16") for idx in idxs]

        th_base = self._print(indexed.base,
                              ndim=len(idxs), **kwargs)
        var_map = {var.discrete: var
                   for var
                   in [*self._system.dependent_variables,
                       *self._system.independent_variables]}
        var = var_map[indexed.base]
        if isinstance(var, DependentVariable):
            ranges = [(self._print(ivar.idx.lower, dtypes=self._dtypes,
                                   broadcastables=self._broadcast),
                       self._print(ivar.idx.upper, dtypes=self._dtypes,
                                   broadcastables=self._broadcast))
                      for ivar in var.independent_variables]
        elif isinstance(var, IndependentVariable):
            ranges = [(self._print(var.idx.lower, dtypes=self._dtypes,
                                   broadcastables=self._broadcast),
                       self._print(var.idx.upper, dtypes=self._dtypes,
                                   broadcastables=self._broadcast))]
        idxs = [tt.clip(idx, *range) for idx, range in zip(idxs, ranges)]

        return th_base[tuple(idxs)]
