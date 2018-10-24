#!/usr/bin/env python
# coding=utf-8

import logging
import re
import string
import typing
from functools import partial, reduce
from itertools import chain, product
from operator import mul, and_
from queue import Queue

import attr
import numpy as np
from more_itertools import unique_everseen
from sympy import (
    Expr,
    N,
    differentiate_finite,
    And,
    Derivative,
    Dummy,
    Eq,
    Function,
    Ge,
    Idx,
    Indexed,
    IndexedBase,
    Le,
    Symbol,
    Wild,
    oo,
    solve,
    sympify,
    Min,
    Max,
)
from sympy.logic.boolalg import BooleanTrue

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def _convert_pde_list(pdes):
    if isinstance(pdes, str):
        return [pdes]
    else:
        return pdes


def _apply_centered_scheme(order, ivar, deriv, accuracy, fdiff):
    n = (order + 1) // 2 + accuracy - 2
    points = [ivar.symbol + i * ivar.step for i in range(-n, n + 1)]
    discretized_deriv = deriv.as_finite_difference(points=points, wrt=ivar.symbol)
    fdiff = fdiff.replace(deriv, discretized_deriv)
    return fdiff


def _apply_right_scheme(order, ivar, deriv, accuracy, fdiff):
    n = accuracy + order
    points = [ivar.symbol + i * ivar.step for i in range(0, n)]
    discretized_deriv = deriv.as_finite_difference(points=points, wrt=ivar.symbol)
    fdiff = fdiff.replace(deriv, discretized_deriv)
    return fdiff


def _apply_left_scheme(order, ivar, deriv, accuracy, fdiff):
    n = accuracy + order
    points = [ivar.symbol + i * ivar.step for i in range(-(n - 1), 1)]
    discretized_deriv = deriv.as_finite_difference(points=points, wrt=ivar.symbol)
    fdiff = fdiff.replace(deriv, discretized_deriv)
    return fdiff


def _partial_derivative(expr, symbolic_independent_variables):
    # proxy function that can be easily curried (with functools.partial)
    return Derivative(expr, *symbolic_independent_variables)


def _build_sympy_namespace(
    equation, independent_variables, dependent_variables, parameters
):
    """ Check the equation, find all the derivative in Euler notation
    (see https://en.wikipedia.org/wiki/Notation_for_differentiation#Euler's_notation)
    the way that dxxU will be equal to Derivative(U(x), x, x).
    All the derivative found that way are add to a subsitution rule as dict and
    applied when the equation is sympified.
    """

    # look at all the dxxU, dxyV... and dx(...), dxy(...) and so on in the equation
    spatial_derivative_re = re.compile(
        r"d(?P<derargs>\w+?)(?:(?P<depder>(?:%s)+)|\((?P<inder>.*?)\))"
        % "|".join([var.name for var in chain(dependent_variables, parameters)])
    )
    spatial_derivatives = spatial_derivative_re.findall(str(equation))

    # they can be derivatives inside the dx(...), we check it until there is no more
    queue = Queue()
    [queue.put(sder[2]) for sder in spatial_derivatives if sder[2]]
    while not queue.empty():
        inside_derivative = queue.get()
        new_derivatives = spatial_derivative_re.findall(inside_derivative)
        [queue.put(sder[2]) for sder in new_derivatives if sder[2]]
        spatial_derivatives.extend(new_derivatives)
    # The sympy namespace is built with...
    namespace = dict()
    # All the independent variables
    namespace.update({var.name: var.symbol for var in independent_variables})
    # All the dependent variables and the parameters
    namespace.update(
        {
            var.name: (
                var.symbol(*[ivar.symbol for ivar in var.independent_variables])
                if var.independent_variables
                else var.symbol
            )
            for var in chain(dependent_variables, parameters)
        }
    )
    # All the harversted derivatives
    for ivar, dvar, _ in spatial_derivatives:
        if dvar:
            namespace["d%s%s" % (ivar, dvar)] = _partial_derivative(
                namespace[dvar], ivar
            )
        else:
            namespace["d%s" % ivar] = partial(
                _partial_derivative, symbolic_independent_variables=ivar
            )
    return namespace


def get_domains(indexed, mapper):
    dvar = mapper[str(indexed.args[0])]
    coords = extract_coord(indexed, mapper)
    return dvar.domains(*[coords[ivar.name] for ivar in dvar.ivars])


def get_distances(indexed, mapper):
    dvar = mapper[str(indexed.args[0])]
    coords = extract_coord(indexed, mapper)
    return [ivar.distance_from_domain(coords[ivar.name]) for ivar in dvar.ivars]


def extract_coord(indexed, mapper):
    dvar = mapper[str(indexed.args[0])]
    keys = dvar.i_names
    coords = indexed.args[1:]
    return dict(zip(keys, coords))


def list_coords(domain):
    available_coords = {}
    for ivar, (left_cond, right_cond) in domain.items():
        coords = []
        coords.append(ivar.idx)
        coords.extend(np.arange(ivar.idx.lower, left_cond.rhs))
        coords.extend(
            np.arange(right_cond.rhs - ivar.N + 1, ivar.idx.upper - ivar.N + 1) + ivar.N
        )
        available_coords[ivar] = coords
    return available_coords


def list_conditions(domain):
    available_conds = {}
    for ivar, (left_cond, right_cond) in domain.items():
        conds = []
        conds.append(left_cond & right_cond)
        conds.extend(
            [Eq(ivar.idx, coord) for coord in np.arange(ivar.idx.lower, left_cond.rhs)]
        )
        conds.extend(
            [
                Eq(ivar.idx, coord)
                for coord in np.arange(
                    right_cond.rhs - ivar.N + 1, ivar.idx.upper - ivar.N + 1
                )
                + ivar.N
            ]
        )
        available_conds[ivar] = conds
    return available_conds


def analyse_local_eq(expr, node, mapper):
    local_eq = expr.subs(
        {ivar.idx: coord for ivar, coord in zip(self.dvar.ivars, node)}
    )
    indexed = local_eq.atoms(Indexed)
    info = []
    for idx in indexed:
        domains = get_domains(idx, mapper)
        distances = get_distances(idx, mapper)
        if not all([domain == "bulk" for domain in domains]):
            info.append((idx, domains, distances))
    return info


def extract_bounds(indexed, mapper):
    dvar = mapper[str(indexed.args[0])]
    ivars = dvar.ivars
    coords = extract_coord(indexed, mapper)
    bounds = {}
    for coord_value, ivar in zip(coords.values(), ivars):
        bounds[ivar] = (
            solve(coord_value - ivar.idx.lower, ivar.idx)[0],
            solve(coord_value - ivar.idx.upper, ivar.idx)[0],
        )
    return bounds


def bounds_to_conds(bounds):
    ivars = set(chain(*[bound.keys() for bound in bounds]))
    conds = {}
    for ivar in ivars:
        lefts, rights = zip(*[bound[ivar] for bound in bounds])
        conds[ivar] = max(lefts), min(rights)
    return {
        ivar: (ivar.idx >= conds[ivar][0], ivar.idx <= conds[ivar][1])
        for ivar, cond in conds.items()
    }

def extract_outside_variables(idxs, mapper):
    outside_variables = []
    for idx in idxs:
        domains = get_domains(idx, mapper)
        distances = get_distances(idx, mapper)
        if not all([domain == "bulk" for domain in domains]):
            outside_variables.append((idx, domains, distances))
    return outside_variables

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
    def symbol(self):
        return Symbol(self.name)

    @property
    def discrete(self):
        return IndexedBase("%s_i" % self.name)

    @property
    def N(self):
        return Symbol("N_%s" % self.name, integer=True)

    @property
    def bound(self):
        return (0, self.N - 1)

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
        if (not is_left and not is_right):
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
        return tuple([ivar.is_in_bulk(coord) for ivar, coord in zip(self.ivars, coords)])

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


@attr.s
class PDEquation:
    equation = attr.ib(type=str)
    dependent_variables = attr.ib(
        type=typing.Tuple[DependentVariable], converter=_convert_depvar_list
    )
    parameters = attr.ib(
        type=typing.Tuple[DependentVariable], converter=_convert_depvar_list, default=[]
    )
    independent_variables = attr.ib(
        type=typing.Tuple[IndependentVariable],
        default=[],
        converter=_convert_indepvar_list,
        repr=False,
    )
    auxiliary_definitions = attr.ib(type=dict, default={})
    scheme = attr.ib(type=str, default="centered")
    accuracy_order = attr.ib(type=int, default=2)
    symbolic_equation = attr.ib(init=False, repr=False)
    fdiff = attr.ib(init=False, repr=False)
    raw = attr.ib(type=bool, default=False, repr=False)

    @scheme.validator
    def check_scheme(self, attrs, scheme):
        return scheme in ["left", "right", "centered"]

    def __attrs_post_init__(self):
        if self.scheme == "centered":
            # For the same number of point, the centered scheme is more accurate
            self.accuracy_order = self.accuracy_order + self.accuracy_order % 2
        for aux_key, aux_value in self.auxiliary_definitions.items():
            # substitute the auxiliary definition as string
            # FIXME : obviously not stable, should be replace as sympy expr
            self.equation = self.equation.replace(aux_key, aux_value)
        if self.raw:
            # For "raw" equations already in discretized form as periodic bdc
            self.fdiff = sympify(
                self.equation,
                locals={dvar.name: dvar.discrete for dvar in self.dependent_variables},
            )
            return
        self._t = Symbol("t")
        self._complete_independent_vars()
        self._fill_incomplete_dependent_vars()
        self._sympy_namespace = _build_sympy_namespace(
            self.equation,
            self.independent_variables,
            self.dependent_variables,
            self.parameters,
        )
        self._sympify_equation()
        self._as_finite_diff()

    def _fill_incomplete_dependent_vars(self):
        """fill every dependent variables that lack information on
        independent variables with the global independent variables
        """
        for i, dependent_variable in enumerate(self.dependent_variables):
            if not dependent_variable.independent_variables:
                object.__setattr__(
                    self.dependent_variables[i],
                    "independent_variables",
                    self.independent_variables,
                )

    def _complete_independent_vars(self):
        """if independent variables is not set, extract them from
        the dependent variables. If not set in dependent variables,
        1D resolution with "x" as independent variable is assumed.
        """
        harvested_indep_vars = list(
            chain(
                *[
                    dep_var.independent_variables
                    for dep_var in self.dependent_variables
                    if dep_var.independent_variables is not None
                ]
            )
        )
        if not self.independent_variables and harvested_indep_vars:
            self.independent_variables = harvested_indep_vars
        elif not self.independent_variables and not harvested_indep_vars:
            self.independent_variables = _convert_indepvar_list(["x"])
        else:
            self.independent_variables = tuple(
                [
                    *self.independent_variables,
                    *[
                        var
                        for var in harvested_indep_vars
                        if var not in self.independent_variables
                    ],
                ]
            )
        self.independent_variables = list(unique_everseen(self.independent_variables))

    def _sympify_equation(self):
        self.symbolic_equation = sympify(self.equation, locals=self._sympy_namespace)
        for dvar in self.dependent_variables:
            self.symbolic_equation = self.symbolic_equation.subs(dvar.name, dvar.symbol)
        self.symbolic_equation = self.symbolic_equation.doit()

    def _as_finite_diff(self):
        def is_wrt(ivar_deriv):
            ivar, deriv = ivar_deriv
            return Symbol(str(ivar)) in deriv.args[1:]

        fdiff = self.symbolic_equation
        while fdiff.atoms(Derivative):
            deriv_product = product(self.independent_variables, fdiff.atoms(Derivative))
            deriv_product = filter(is_wrt, deriv_product)
            for ivar, deriv in product(
                self.independent_variables, fdiff.atoms(Derivative)
            ):
                wrts = {}
                for wrt in deriv.args[1:]:
                    if isinstance(wrt, Symbol):
                        wrts[wrt] = 1
                    else:
                        wrts[wrt[0]] = wrt[1]

                order = wrts.get(ivar.symbol, 0)
                if self.scheme == "centered":
                    fdiff = _apply_centered_scheme(
                        order, ivar, deriv, self.accuracy_order, fdiff
                    )
                elif self.scheme == "right":
                    fdiff = _apply_right_scheme(
                        order, ivar, deriv, self.accuracy_order, fdiff
                    )
                elif self.scheme == "left":
                    fdiff = _apply_left_scheme(
                        order, ivar, deriv, self.accuracy_order, fdiff
                    )

        def upwind(velocity, dvar, ivar, accuracy=1):
            def left_deriv(ivar, dvar):
                deriv = Derivative(dvar, (ivar, 1))
                n = accuracy + 1
                if accuracy < 3:
                    points = [ivar.symbol + i * ivar.step for i in range(-(n - 1), 1)]
                elif accuracy == 3:
                    points = [ivar.symbol + i * ivar.step for i in range(-(n - 2), 2)]
                else:
                    raise NotImplementedError("Upwind is only available for n <= 3")
                discretized_deriv = deriv.as_finite_difference(
                    points=points, wrt=ivar.symbol
                )
                return discretized_deriv

            def right_deriv(ivar, dvar):
                deriv = Derivative(dvar, (ivar, 1))
                n = accuracy + 1
                if accuracy < 3:
                    points = [ivar.symbol + i * ivar.step for i in range(0, n)]
                elif accuracy == 3:
                    points = [ivar.symbol + i * ivar.step for i in range(-1, n - 1)]
                else:
                    raise NotImplementedError("Upwind is only available for n <= 3")
                discretized_deriv = deriv.as_finite_difference(
                    points=points, wrt=ivar.symbol
                )
                return discretized_deriv

            ap = Max(velocity, 0)
            am = Min(velocity, 0)

            ivar = IndependentVariable(str(ivar))

            deriv_left = left_deriv(ivar, dvar)
            deriv_right = right_deriv(ivar, dvar)
            discretized_deriv = am * deriv_left + ap * deriv_right
            return discretized_deriv

        fdiff = fdiff.replace(Function("upwind"), upwind)

        for ivar in self.independent_variables:
            a = Wild("a", exclude=[ivar.step, ivar.symbol, 0])
            fdiff = fdiff.replace(ivar.symbol + a * ivar.step, ivar.idx + a)

        for var in chain(self.dependent_variables, self.parameters):

            def replacement(*args):
                return var.discrete[args]

            if var.independent_variables:
                fdiff = fdiff.replace(var.symbol, replacement)

        for indexed in fdiff.atoms(Indexed):
            new_indexed = indexed.subs(
                {ivar.symbol: ivar.idx for ivar in self.independent_variables}
            )
            fdiff = fdiff.subs(indexed, new_indexed)

        fdiff = fdiff.subs(
            {
                ivar.symbol: ivar.discrete[ivar.idx]
                for ivar in self.independent_variables
            }
        )

        self.fdiff = fdiff

    def __str__(self):
        return self.__repr__()


@attr.s(auto_attribs=True)
class Node:
    eq: Expr
    dvar: DependentVariable
    coords: tuple
    conds: Expr
    mapper: dict
    available_boundaries: dict
    outside_variables = None
    linked_nodes = None

    def __attrs_post_init__(self):
        self.local_eq = self.eq.subs(
            {ivar.idx: coord for ivar, coord in zip(self.dvar.ivars, self.coords)}
        )
        self.linked_nodes = []
        while self.outside_variables:
            self.apply_boundary()

    @property
    def outside_variables(self):
        indexed = self.local_eq.atoms(Indexed)
        return extract_outside_variables(indexed, self.mapper)

    def apply_boundary(self):
        def get_proper_boundary(boundaries, domains):
            for ivar, domain in zip(dvar.ivars, domains):
                left_bdc, right_bdc = boundaries[ivar]
                if domain == "left":
                    yield left_bdc.fdiff
                if domain == "right":
                    yield right_bdc.fdiff

        subs = {}
        for idx, domains, distances in self.outside_variables:
            dvar = self.mapper[str(idx.args[0])]
            bdcs = list(get_proper_boundary(self.available_boundaries[dvar], domains))
            bdcs = [
                bdc.subs(
                    {ivar.idx: coord for ivar, coord in zip(dvar.ivars, idx.args[1:])}
                )
                for bdc in bdcs
            ]
            subs.update(solve(bdcs, idx))
        self.local_eq = self.local_eq.subs(subs)


@attr.s
class PDESys:
    evolution_equations = attr.ib(type=typing.List[str], converter=_convert_pde_list)
    dependent_variables = attr.ib(
        type=typing.List[DependentVariable], converter=_convert_depvar_list
    )
    parameters = attr.ib(
        type=typing.List[DependentVariable], converter=_convert_depvar_list
    )
    independent_variables = attr.ib(
        type=typing.List[IndependentVariable],
        default=[],
        converter=_convert_indepvar_list,
        repr=False,
    )
    boundary_conditions = attr.ib(type=dict, default=None)
    auxiliary_definitions = attr.ib(type=dict, default=None)
    _domains = attr.ib(type=dict, default=None, init=False, repr=False)

    def _compute_domain(self):
        self._domains = []
        self._unknown_nodes = []
        for dvar, pde in zip(self.dependent_variables, self):
            bounds = [
                extract_bounds(indexed, self.dependent_dict)
                for indexed in pde.fdiff.atoms(Indexed)
            ]
            self._domains.append(bounds_to_conds(bounds))

    def _coerce_equations(self):
        self.evolution_equations = [
            PDEquation(
                eq,
                self.dependent_variables,
                self.parameters,
                auxiliary_definitions=self.auxiliary_definitions,
            )
            for eq in self.evolution_equations.copy()
        ]
        self.independent_variables = sorted(
            set(
                chain(
                    *[
                        eq.independent_variables
                        for eq in self.evolution_equations.copy()
                    ]
                )
            )
        )

    def _ensure_bdc(self):
        def not_in_iter(iter, dvar):
            return dvar.name not in iter

        def in_iter(iter, dvar):
            return dvar.name in iter

        if self.boundary_conditions == "periodic":
            self.boundary_conditions = dict()
            for dvar in chain(self.dependent_variables, self.parameters):
                self.boundary_conditions[dvar.name] = dict()
                for ivar in self.independent_variables:
                    self.boundary_conditions[dvar.name][ivar.name] = "periodic"

        if self.boundary_conditions == "noflux":
            self.boundary_conditions = dict()

        for dvar in filter(
            partial(not_in_iter, self.boundary_conditions),
            chain(self.dependent_variables, self.parameters),
        ):
            self.boundary_conditions[dvar] = {
                ivar.name: ["d%s%s" % (ivar.name, dvar.name)] * 2
                for ivar in dvar.independent_variables
            }
        for dvar in filter(
            partial(in_iter, self.boundary_conditions),
            chain(self.dependent_variables, self.parameters),
        ):
            self.boundary_conditions[dvar] = self.boundary_conditions.pop(dvar.name)
            for ivar in filter(
                partial(not_in_iter, self.boundary_conditions[dvar]),
                dvar.independent_variables,
            ):
                self.boundary_conditions[dvar][ivar.name] = [
                    "d%s%s" % (ivar.name, dvar.name)
                ] * 2
        for dvar, bdc in self.boundary_conditions.items():
            keys = list(bdc.keys())
            new_keys = _convert_indepvar_list(keys)
            for new_key, old_key in zip(new_keys, keys):
                bdc[new_key] = bdc.pop(old_key)
            for ivar, eqs in bdc.items():
                if eqs == "periodic":
                    left_cond = dvar.discrete_i - dvar.discrete_i.subs(
                        ivar.idx, ivar.idx + ivar.N
                    )
                    right_cond = dvar.discrete_i - dvar.discrete_i.subs(
                        ivar.idx, ivar.idx - ivar.N
                    )
                    bdc[ivar] = (
                        PDEquation(
                            left_cond,
                            dependent_variables=self.dependent_variables,
                            independent_variables=dvar.independent_variables,
                            raw=True,
                        ),
                        PDEquation(
                            right_cond,
                            dependent_variables=self.dependent_variables,
                            independent_variables=dvar.independent_variables,
                            raw=True,
                        ),
                    )
                else:
                    bdc[ivar] = [
                        PDEquation(
                            eq if eq else "d%s%s" % (ivar.name, dvar.name),
                            dependent_variables=self.dependent_variables,
                            independent_variables=dvar.independent_variables,
                            scheme=scheme,
                        )
                        for scheme, eq in zip(["right", "left"], eqs)
                    ]
            self.boundary_conditions[dvar] = bdc

    def _build_system(self):
        self.nodes = []
        self._system = []
        for dvar, domain, pde in zip(self.dependent_variables, self._domains, self):
            nodes = []
            coords_ = list_coords(domain)
            conds = list_conditions(domain)
            all_coords = list(product(*[coords_[ivar] for ivar in dvar.ivars]))
            all_conds = list(product(*[conds[ivar] for ivar in dvar.ivars]))
            for coords, conds in zip(all_coords, all_conds):
                conds = reduce(and_, conds)
                node = Node(eq=pde.fdiff, dvar=dvar, coords=coords, conds=conds,
                            mapper=self.dependent_dict,
                            available_boundaries=self.boundary_conditions)
                nodes.append(node)
            self.nodes.append(nodes)
            self._system.append([(node.conds, node.local_eq) for node in nodes])

    def _get_shapes(self):
        self.pivot = None

        gridinfo = dict(
            zip(
                self.independent_variables,
                [ivar.N for ivar in self.independent_variables],
            )
        )

        shapes = []
        for depvar in self.dependent_variables:
            gridshape = [
                (gridinfo[ivar] if ivar in depvar.independent_variables else 1)
                for ivar in self.independent_variables
            ]
            shapes.append(tuple(gridshape))
        sizes = [reduce(mul, shape) for shape in shapes]
        self.size = sum(sizes)

        self.shapes = dict(zip(self.dependent_variables, shapes))
        self.sizes = dict(zip(self.dependent_variables, sizes))

    def __attrs_post_init__(self):
        logging.info("processing pde system")
        self._system = []
        if self.boundary_conditions is None:
            self.boundary_conditions = dict()
        if self.auxiliary_definitions is None:
            self.auxiliary_definitions = dict()
        logging.info("coerce equations...")
        self._t = Symbol("t")
        self._coerce_equations()
        logging.info("compute domain...")
        self._compute_domain()
        logging.info("ensure bdc...")
        self._ensure_bdc()
        logging.info("build system...")
        self._get_shapes()
        logging.info("get shapes...")
        self._build_system()
        logging.info("done")

    @property
    def dependent_dict(self):
        return {depvar.name: depvar for depvar in self.dependent_variables}

    @property
    def independent_dict(self):
        return {depvar.name: depvar.ivars for depvar in self.dependent_variables}

    @property
    def equation_dict(self):
        return {
            depvar.name: equation
            for depvar, equation in zip(
                self.dependent_variables, self.evolution_equations
            )
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.evolution_equations[key]
        if isinstance(key, str):
            return self.equation_dict[key]
        raise KeyError(key)
