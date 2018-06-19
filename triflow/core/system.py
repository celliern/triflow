#!/usr/bin/env python
# coding=utf8

import re
import string
import typing
from functools import partial, reduce
from itertools import chain, product
from operator import mul
from queue import Queue

import attr
import numpy as np
import sympy as sp
from more_itertools import unique_everseen
from sympy import (Derivative, Function, Idx, Indexed, IndexedBase, Symbol,
                   Wild, solve, sympify)


def _convert_pde_list(pdes):
    if isinstance(pdes, str):
        return [pdes]
    else:
        return pdes


def is_in_bulk(indexed, ivars):
    ivar_idxs_value = indexed.args[1:]
    is_inside = True
    for ivar, value in zip(ivars, ivar_idxs_value):
        if ivar.idx in value.atoms(Idx):
            is_inside *= True
            continue

        left_cond = (value >= ivar.idx.lower)
        if not isinstance(left_cond, bool):
            left_cond = left_cond.subs(ivar.N, sp.oo)

        right_cond = (value <= ivar.idx.upper)
        if not isinstance(right_cond, bool):
            right_cond = right_cond.subs(ivar.N, sp.oo)

        is_inside *= bool(left_cond & right_cond)
    return bool(is_inside)


def get_domain(indexed, ivars):
    ivar_idxs_value = indexed.args[1:]
    where = []
    for ivar, value in zip(ivars, ivar_idxs_value):
        if ivar.idx in value.atoms(Idx):
            where.append("bulk")
        elif (value < ivar.idx.lower).subs(ivar.N, sp.oo):
            where.append("left")
        elif (value > ivar.idx.upper).subs(ivar.N, sp.oo):
            where.append("right")
        else:
            where.append("bulk")
    return where


def compute_bdc_on_ghost_node(indexed, ivars, bdcs):
    domains = get_domain(indexed, ivars)
    ghost_node = indexed.args[1:]
    for ivar, node, domain in zip(ivars, ghost_node, domains):
        if domain == "left":
            yield bdcs[ivar][0].fdiff_equation.subs(
                {ivar.idx: coord for ivar, coord in zip(ivars, ghost_node)})
        if domain == "right":
            yield bdcs[ivar][1].fdiff_equation.subs(
                {ivar.idx: coord for ivar, coord in zip(ivars, ghost_node)})


@attr.s(frozen=True, repr=False, hash=False)
class IndependentVariable:
    name = attr.ib(type=str)

    @name.validator
    def check(self, attrs, value):
        if value not in string.ascii_letters or len(value) > 1:
            raise ValueError("independant variables have to be 1 char "
                             "lenght and be an ascii letter "
                             "(is \"%s\")" % value)

    @property
    def symbol(self):
        return Symbol(self.name)

    @property
    def discrete(self):
        return IndexedBase("%s_i" % self.name)

    @property
    def N(self):
        return Symbol("N_%s" % self.name, integer=True)

    @property
    def idx(self):
        return Idx("%s_idx" % self.name, self.N)

    @property
    def step(self):
        return Symbol("d%s" % self.name)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.name)


_dependent_var_re = re.compile(r"^(?P<dvar>\w+)(?:\((?P<dvararg>[^\)]*)\))?$")


@attr.s(frozen=True, repr=False, hash=False)
class DependentVariable:
    name = attr.ib(type=str)
    independent_variables = attr.ib(init=False,
                                    type=typing.Tuple[IndependentVariable])

    def __attrs_post_init__(self):
        name, independent_variables = DependentVariable._convert(self.name)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "independent_variables", independent_variables)

    @staticmethod
    def _convert(var):
        match = _dependent_var_re.match(var)
        if not match:
            raise ValueError("Dependent variable not properly formatted.")

        depvar, indepvars = _dependent_var_re.findall(var)[0]
        return depvar, tuple([IndependentVariable(indepvar.strip())
                              for indepvar in indepvars.split(',')
                              if indepvar != ''])

    @property
    def symbol(self):
        return Function(self.name)

    @property
    def discrete(self):
        return IndexedBase(self.name)

    @property
    def discrete_i(self):
        return self.discrete[
            tuple([ivar.idx for ivar in self.independent_variables])]

    def __repr__(self):
        return "{}({})".format(
            self.name, ", ".join(map(str, self.independent_variables)))

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__str__())


def _convert_depvar_list(dependent_variables):
    def convert_depvar(dependent_variable):
        if isinstance(dependent_variable, str):
            return DependentVariable(dependent_variable)
        if isinstance(dependent_variable, DependentVariable):
            return dependent_variable
        raise ValueError("dependent var should be string or DependentVariable")

    if not isinstance(dependent_variables, (list, tuple)):
        return [convert_depvar(dependent_variables)]
    else:
        return tuple(list(map(convert_depvar, dependent_variables)))


def _convert_indepvar_list(independent_variables):
    def convert_indepvar(independent_variable):
        if isinstance(independent_variable, str):
            return IndependentVariable(independent_variable)
        if isinstance(independent_variable, IndependentVariable):
            return independent_variable
        raise ValueError(
            "dependent var should be string or IndependentVariable")

    if not isinstance(independent_variables, (list, tuple)):
        return [convert_indepvar(independent_variables)]
    else:
        return tuple(list(map(convert_indepvar, independent_variables)))


@attr.s(repr=False)
class PDEquation:
    equation = attr.ib(type=str)
    dependent_variables = attr.ib(type=typing.Tuple[DependentVariable],
                                  converter=_convert_depvar_list)
    independent_variables = attr.ib(type=typing.Tuple[IndependentVariable],
                                    default=[],
                                    converter=_convert_indepvar_list)
    auxiliary_definitions = attr.ib(type=dict,
                                    default={})
    scheme = attr.ib(type=str, default="centered")
    accuracy_order = attr.ib(type=int, default=2)
    symbolic_equation = attr.ib(init=False)
    fdiff_equation = attr.ib(init=False)
    raw = attr.ib(type=bool, default=False)

    @scheme.validator
    def check_scheme(self, attrs, scheme):
        return scheme in ["left", "right", "centered"]

    def __attrs_post_init__(self):
        if self.scheme == "centered":
            self.accuracy_order = (self.accuracy_order +
                                   self.accuracy_order % 2)
        for aux_key, aux_value in self.auxiliary_definitions.items():
            self.equation = self.equation.replace(aux_key, aux_value)
        if self.raw:
            self.fdiff_equation = sympify(self.equation,
                                          locals={dvar.name: dvar.discrete
                                                  for dvar
                                                  in self.dependent_variables})
            return
        self._complete_independent_vars()
        self._fill_incomplete_dependent_vars()
        self._build_sympify_namespace()
        self._sympify_equation()
        self._as_finite_diff()
        self._extract_bounds()

    def _fill_incomplete_dependent_vars(self):
        """fill every dependent variables that lack information on
        independent variables with the global independent variables
        """
        for i, dependent_variable in enumerate(self.dependent_variables):
            if not dependent_variable.independent_variables:
                object.__setattr__(self.dependent_variables[i],
                                   "independent_variables",
                                   self.independent_variables)

    def _complete_independent_vars(self):
        """if independent variables is not set, extract them from
        the dependent variables.
        """
        harvested_indep_vars = list(
            chain(*[dep_var.independent_variables
                    for dep_var in self.dependent_variables
                    if dep_var.independent_variables is not None]))
        if not self.independent_variables and harvested_indep_vars:
            self.independent_variables = harvested_indep_vars
        elif (not self.independent_variables and
              not harvested_indep_vars):
            self.independent_variables = _convert_indepvar_list(["x"])
        else:
            self.independent_variables = tuple(
                [*self.independent_variables,
                 *[var for var in harvested_indep_vars
                   if var not in self.independent_variables]])
        self.independent_variables = list(
            unique_everseen(self.independent_variables))

    def _build_sympify_namespace(self):
        def partial_derivative(expr, symbolic_independent_variables):
            return Derivative(expr, *symbolic_independent_variables)
        spatial_derivative_re = re.compile(
            r"d(?P<derargs>\w+?)(?:(?P<depder>[%s]+)|\((?P<inder>.*?)\))" %
            ','.join([var.name for var in self.dependent_variables]))
        spatial_derivative = spatial_derivative_re.findall(str(self.equation))

        namespace = dict()
        namespace.update({var.name: var.symbol
                          for var in self.independent_variables})
        namespace.update({var.name: var.symbol(*[ivar.symbol
                                                 for ivar
                                                 in var.independent_variables])
                          for var in self.dependent_variables})
        for ivar, dvar, inside in spatial_derivative:
            if dvar:
                namespace['d%s%s' % (ivar, dvar)] = partial_derivative(
                    namespace[dvar], ivar)
            else:
                namespace['d%s' % ivar] = partial(
                    partial_derivative, symbolic_independent_variables=ivar)
        self._sympy_namespace = namespace

    def _sympify_equation(self):
        self.symbolic_equation = sympify(self.equation,
                                         locals=self._sympy_namespace)
        for dvar in self.dependent_variables:
            self.symbolic_equation = self.symbolic_equation.subs(dvar.name,
                                                                 dvar.symbol)
        self.symbolic_equation = self.symbolic_equation.doit()

    def _as_finite_diff(self):
        fdiff_equation = self.symbolic_equation
        for ivar in self.independent_variables:
            for deriv in fdiff_equation.atoms(Derivative):
                order = deriv.args[1:].count(ivar.symbol)
                if self.scheme == "centered":
                    n = (order + 1) // 2 + self.accuracy_order - 2
                    fdiff_equation = fdiff_equation.replace(
                        deriv, deriv.as_finite_difference(
                            points=[ivar.symbol + i * ivar.step
                                    for i in range(-n, n+1)],
                            wrt=ivar.symbol))
                elif self.scheme == "right":
                    n = self.accuracy_order + order
                    fdiff_equation = fdiff_equation.replace(
                        deriv, deriv.as_finite_difference(
                            points=[ivar.symbol + i * ivar.step
                                    for i in range(0, n)],
                            wrt=ivar.symbol))
                elif self.scheme == "left":
                    n = self.accuracy_order + order
                    fdiff_equation = fdiff_equation.replace(
                        deriv, deriv.as_finite_difference(
                            points=[ivar.symbol + i * ivar.step
                                    for i in range(-(n-1), 1)],
                            wrt=ivar.symbol))

        for ivar in self.independent_variables:
            a = Wild("a", exclude=[ivar.step, ivar.symbol, 0])
            fdiff_equation = fdiff_equation.replace(
                ivar.symbol + a * ivar.step, ivar.idx + a)

        for var in self.dependent_variables:
            def replacement(*args):
                return var.discrete[args]
            fdiff_equation = fdiff_equation.replace(var.symbol, replacement)

        for indexed in fdiff_equation.atoms(Indexed):
            new_indexed = indexed.subs({ivar.symbol: ivar.idx
                                        for ivar
                                        in self.independent_variables})
            fdiff_equation = fdiff_equation.subs(indexed, new_indexed)

        fdiff_equation = fdiff_equation.subs({ivar.symbol:
                                              ivar.discrete[ivar.idx]
                                              for ivar
                                              in self.independent_variables})

        self.fdiff_equation = fdiff_equation

    def _extract_bounds(self):
        for indexed in self.fdiff_equation.atoms(Indexed):
            pass

    def __repr__(self):
        return str(self.equation)

    def __str__(self):
        return self.__repr__()


@attr.s
class PDESys:
    evolution_equations = attr.ib(type=typing.List[str],
                                  converter=_convert_pde_list)
    dependent_variables = attr.ib(type=typing.List[DependentVariable],
                                  converter=_convert_depvar_list)
    independent_variables = attr.ib(type=typing.List[IndependentVariable],
                                    default=[],
                                    converter=_convert_indepvar_list)
    boundary_conditions = attr.ib(type=dict, default={})
    auxiliary_definitions = attr.ib(type=dict, default={})
    _domains = attr.ib(type=dict, default=None, init=False)

    def _compute_domain(self):
        self._domains = []
        for dvar, pde in zip(self.dependent_variables, self):
            ivars = dvar.independent_variables
            eq = pde.fdiff_equation
            domain = {}
            for i, ivar in enumerate(ivars, 1):
                lower_conds = [(indexed.args[i] - ivar.idx.lower)
                               .subs(ivar.idx, ivar.symbol)
                               for indexed in eq.atoms(Indexed)
                               if indexed.args[0] not in [
                                   ivar.discrete
                                   for ivar
                                   in dvar.independent_variables]]
                upper_conds = [(indexed.args[i] - ivar.idx.upper)
                               .subs(ivar.idx, ivar.symbol)
                               for indexed in eq.atoms(Indexed)
                               if indexed.args[0] not in [
                                   ivar.discrete
                                   for ivar
                                   in dvar.independent_variables]]
                cond = (sp.Ge(ivar.idx, max([solve(cond, ivar.symbol)[0]
                                             for cond in lower_conds])),
                        sp.Le(ivar.idx, min([solve(cond, ivar.symbol)[0]
                                             for cond in upper_conds])))
                domain[ivar] = cond
            self._domains.append(domain)
        self._unknown_nodes = []
        for domain in self._domains:
            unodes = {}
            for ivar, (lower_cond, upper_cond) in domain.items():
                if isinstance(lower_cond, (bool, sp.boolalg.BooleanTrue)):
                    left_unodes = []
                else:
                    left_unodes = np.arange(ivar.idx.lower,
                                            lower_cond.rhs).tolist()
                if isinstance(upper_cond, (bool, sp.boolalg.BooleanTrue)):
                    right_unodes = []
                else:
                    right_unodes = np.arange(ivar.idx.upper,
                                             upper_cond.rhs,
                                             -1).tolist()
                unodes[ivar] = (left_unodes, right_unodes)
            self._unknown_nodes.append(unodes)

    def _coerce_equations(self):
        self.evolution_equations = [PDEquation(
            eq, self.dependent_variables,
            auxiliary_definitions=self.auxiliary_definitions)
            for eq in self.evolution_equations]
        self.independent_variables = sorted(
            set(chain(*[eq.independent_variables
                        for eq in self.evolution_equations]))
        )

    def _ensure_bdc(self):

        def not_in_iter(iter, dvar):
            return dvar.name not in iter

        def in_iter(iter, dvar):
            return dvar.name in iter

        for dvar in filter(partial(not_in_iter, self.boundary_conditions),
                           self.dependent_variables):
            self.boundary_conditions[dvar] = {ivar.name:
                                              ["d%s%s" % (ivar.name,
                                                          dvar.name)] * 2
                                              for ivar
                                              in dvar.independent_variables}
        for dvar in filter(partial(in_iter, self.boundary_conditions),
                           self.dependent_variables):
            self.boundary_conditions[dvar] = self.boundary_conditions.pop(
                dvar.name)
            for ivar in filter(partial(not_in_iter,
                                       self.boundary_conditions[dvar]),
                               dvar.independent_variables):
                self.boundary_conditions[dvar][ivar.name] = ["d%s%s" %
                                                             (ivar.name,
                                                              dvar.name)] * 2

        for dvar, bdc in self.boundary_conditions.items():
            keys = list(bdc.keys())
            new_keys = _convert_indepvar_list(keys)
            for new_key, old_key in zip(new_keys, keys):
                bdc[new_key] = bdc.pop(old_key)
            for ivar, eqs in bdc.items():
                if eqs == "periodic":
                    left_cond = (dvar.discrete_i -
                                 dvar.discrete_i.subs(ivar.idx,
                                                      ivar.idx + ivar.N))
                    right_cond = (dvar.discrete_i -
                                  dvar.discrete_i.subs(ivar.idx,
                                                       ivar.idx - ivar.N))
                    bdc[ivar] = (PDEquation(
                        left_cond,
                        dependent_variables=self.dependent_variables,
                        independent_variables=dvar.independent_variables,
                        raw=True),
                        PDEquation(
                        right_cond,
                        dependent_variables=self.dependent_variables,
                        independent_variables=dvar.independent_variables,
                        raw=True))
                else:
                    bdc[ivar] = [PDEquation(
                        eq if eq else "d%s%s" % (ivar.name, dvar.name),
                        dependent_variables=self.dependent_variables,
                        independent_variables=dvar.independent_variables,
                        scheme=scheme)
                        for scheme, eq in zip(["right", "left"], eqs)]
            self.boundary_conditions[dvar] = bdc

    def _build_system(self):
        for eq, sysdomain, dvar in zip(self, self._domains,
                                       self.dependent_variables):
            system = {}
            ivars = self.independent_variables
            _unknown_nodes = {ivar: tuple(zip(*[
                map(set, unode[ivar])
                for dvar, unode in zip(self.dependent_variables,
                                       self._unknown_nodes)
                if ivar in dvar.independent_variables]))
                for ivar in self.independent_variables}
            _unknown_nodes = {key: tuple(map(tuple, [reduce(set.union, bdc)
                                                     for bdc in value]))
                              for key, value in _unknown_nodes.items()}

            idxs = []
            for ivar, unode in _unknown_nodes.items():
                left_nodes, right_nodes = unode
                bulk_nodes = [ivar.idx]
                idx = chain(*[left_nodes, bulk_nodes, right_nodes])
                idxs.append(idx)
            coords_to_compute = list(product(*idxs))

            bdcs = self.boundary_conditions.values()

            for coords in coords_to_compute:
                local_eq = eq.fdiff_equation.subs({ivar.idx: coord
                                                   for ivar, coord
                                                   in zip(ivars, coords)})
                subs_queue = Queue()
                unavailable_vars = [indexed
                                    for indexed
                                    in local_eq.atoms(Indexed)
                                    if not is_in_bulk(indexed, ivars)]
                subs_queue.put(unavailable_vars)

                all_unavailable_vars = set()
                all_available_bdcs = set()
                solved_variables = dict()

                for unavailable_vars in iter(subs_queue.get, []):
                    available_bdcs = chain(
                        *[compute_bdc_on_ghost_node(unavailable_var,
                                                    bdc.keys(),
                                                    bdc)
                          for unavailable_var, bdc
                          in product(unavailable_vars, bdcs)])
                    available_bdcs = [bdc.subs(solved_variables)
                                      for bdc in available_bdcs]
                    for bdc in available_bdcs:
                        subs_queue.put(
                            [indexed for indexed in bdc.atoms(Indexed)
                             if not is_in_bulk(indexed, ivars) and
                             indexed not in unavailable_vars])
                    all_available_bdcs = all_available_bdcs.union(
                        set(available_bdcs))
                    all_unavailable_vars = all_unavailable_vars.union(
                        set(unavailable_vars))

                    solved_variables = solve(all_available_bdcs,
                                             all_unavailable_vars)
                    local_eq = local_eq.subs(solved_variables)

                domain_map = []
                for coord, ivar in zip(coords, dvar.independent_variables):
                    if coord == ivar.idx:
                        domain_map.append(sp.And(*sysdomain[ivar]))
                    else:
                        domain_map.append(sp.Eq(ivar.idx, coord))
                domain_map = sp.And(*domain_map)
                system[domain_map] = local_eq
            self._system.append(system)

    def _get_shapes(self):
        self.pivot = None

        gridinfo = dict(zip(self.independent_variables,
                            [ivar.N for ivar in self.independent_variables]))

        shapes = []
        for depvar in self.dependent_variables:
            gridshape = [(gridinfo[ivar]
                          if ivar
                          in depvar.independent_variables
                          else 1)
                         for ivar
                         in self.independent_variables]
            shapes.append(tuple(gridshape))
        sizes = [reduce(mul, shape) for shape in shapes]
        self.size = sum(sizes)

        self.shapes = dict(zip(self.dependent_variables, shapes))
        self.sizes = dict(zip(self.dependent_variables, sizes))

    def __attrs_post_init__(self):
        self._system = []
        self._coerce_equations()
        self._compute_domain()
        self._get_shapes()
        self._ensure_bdc()
        self._build_system()
        self._piecewise_system = [sp.Piecewise(*[(value, key)
                                                 for key, value
                                                 in system.items()])
                                  for system in self._system]

    @property
    def dependent_dict(self):
        return {depvar.name: depvar for depvar in self.dependent_variables}

    @property
    def equation_dict(self):
        return {depvar.name: equation for
                depvar, equation in zip(self.dependent_variables,
                                        self.evolution_equations)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.evolution_equations[key]
        if isinstance(key, str):
            return self.equation_dict[key]
        raise KeyError(key)
