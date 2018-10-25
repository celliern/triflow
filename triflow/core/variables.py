#!/usr/bin/env python
# coding=utf-8

import re
import string
import typing

import attr

from sympy import Function, Idx, IndexedBase, Symbol


@attr.s(frozen=True, repr=False, hash=False)
class IndependentVariable:
    name = attr.ib(type=str)

    @name.validator
    def check(self, attrs, value):
        if value not in string.ascii_letters or len(value) > 1:
            raise ValueError(
                "independant variables have to be 1 char "
                "lenght and be an ascii letter "
                '(is "%s")' % value
            )

    @property
    def ivars(self):
        return (self,)

    @property
    def i_symbols(self):
        return tuple([ivar.symbol for ivar in self.ivars])

    @property
    def i_names(self):
        return tuple([ivar.name for ivar in self.ivars])

    @property
    def i_steps(self):
        return tuple([ivar.step for ivar in self.ivars])

    @property
    def i_step_values(self):
        return tuple([ivar.step_value for ivar in self.ivars])

    @property
    def i_discs(self):
        return tuple([ivar.discrete for ivar in self.ivars])

    @property
    def i_idxs(self):
        return tuple([ivar.idx for ivar in self.ivars])

    @property
    def i_Ns(self):
        return tuple([ivar.N for ivar in self.ivars])

    @property
    def i_bounds(self):
        return tuple([ivar.bound for ivar in self.ivars])

    @property
    def symbol(self):
        return Symbol(self.name)

    @property
    def discrete(self):
        return IndexedBase(self.name)

    @property
    def N(self):
        return Symbol("N_%s" % self.name, integer=True)

    @property
    def bound(self):
        return (0, self.N - 1)

    def domains(self, *coords):
        return tuple([ivar.domain(coord) for ivar, coord in zip(self.ivars, coords)])

    def domain(self, coord):
        def to_bool(cond):
            try:
                cond = cond.subs(self.N, 500)
            except AttributeError:
                pass
            return cond

        is_left = to_bool(coord < self.idx.lower)
        is_right = to_bool(coord > self.idx.upper)
        # if x_idx is in the coord, it belong to the bulk.
        # Otherwise, it's only if it's neither into the left and the right.
        try:
            if str(self.idx) in map(str, coord.atoms()):
                return "bulk"
        except AttributeError:
            pass
        if not is_left and not is_right:
            return "bulk"
        if is_left:
            return "left"
        if is_right:
            return "right"

    def is_in_bulk(self, coord):
        return True if self.domain(coord) == "bulk" else False

    def distance_from_domain(self, coord):
        if self.domain(coord) == "bulk":
            return 0
        if self.domain(coord) == "left":
            return int(self.idx.lower - coord)
        if self.domain(coord) == "right":
            return int(coord - self.idx.upper)

    @property
    def idx(self):
        return Idx("%s_idx" % self.name, self.N)

    @property
    def step(self):
        return Symbol("d%s" % self.name)

    @property
    def step_value(self):
        return (self.discrete[self.idx.upper] - self.discrete[self.idx.lower]) / (
            self.N - 1
        )

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
    independent_variables = attr.ib(init=False, type=typing.Tuple[IndependentVariable])

    def __attrs_post_init__(self):
        name, independent_variables = DependentVariable._convert(self.name)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "independent_variables", independent_variables)

    @staticmethod
    def _convert(var):
        match = _dependent_var_re.match(var)
        if not match:
            raise ValueError("Dependent variable not properly formatted.")

        depvar, indepvars = _dependent_var_re.findall(var)[0]
        return (
            depvar,
            tuple(
                [
                    IndependentVariable(indepvar.strip())
                    for indepvar in indepvars.split(",")
                    if indepvar != ""
                ]
            ),
        )

    @property
    def ivars(self):
        return self.independent_variables

    @property
    def i_symbols(self):
        return tuple([ivar.symbol for ivar in self.ivars])

    @property
    def i_names(self):
        return tuple([ivar.name for ivar in self.ivars])

    @property
    def i_steps(self):
        return tuple([ivar.step for ivar in self.ivars])

    @property
    def i_step_values(self):
        return tuple([ivar.step_value for ivar in self.ivars])

    @property
    def i_discs(self):
        return tuple([ivar.discrete for ivar in self.ivars])

    @property
    def i_idxs(self):
        return tuple([ivar.idx for ivar in self.ivars])

    @property
    def i_Ns(self):
        return tuple([ivar.N for ivar in self.ivars])

    @property
    def i_bounds(self):
        return tuple([ivar.bound for ivar in self.ivars])

    def domains(self, *coords):
        return tuple([ivar.domain(coord) for ivar, coord in zip(self.ivars, coords)])

    def is_in_bulk(self, *coords):
        return tuple(
            [ivar.is_in_bulk(coord) for ivar, coord in zip(self.ivars, coords)]
        )

    @property
    def symbol(self):
        return Function(self.name) if self.ivars else Symbol(self.name)

    @property
    def discrete(self):
        return IndexedBase(self.name) if self.ivars else Symbol(self.name)

    def __len__(self):
        return len(self.ivars)

    @property
    def discrete_i(self):
        return self.discrete[self.i_idxs] if self.ivars else Symbol(self.name)

    def __repr__(self):
        return "{}{}".format(
            self.name, ("(%s)" % ", ".join(map(str, self.ivars)) if self.ivars else "")
        )

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
        raise ValueError("dependent var should be string or IndependentVariable")

    if not isinstance(independent_variables, (list, tuple)):
        return [convert_indepvar(independent_variables)]
    else:
        return tuple(list(map(convert_indepvar, independent_variables)))
