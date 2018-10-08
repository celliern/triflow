#!/usr/bin/env python
# coding=utf-8

import logging
import re
import string
import typing
from functools import partial, reduce
from itertools import chain, product
from operator import mul
from queue import Queue

import attr
import numpy as np
from more_itertools import unique_everseen
from sympy import (
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


def _filter_relevent_equations(bdc, vars):
    for indexed in bdc.atoms(Indexed):
        if indexed in vars:
            return True
    return False


def _ensure_bool(cond, ivars):
    if isinstance(cond, bool):
        return cond
    else:
        cond = cond.subs({ivar.N: oo for ivar in ivars})
        return bool(cond)


def _is_in_bulk(indexed, ivars, all_ivars):
    ivar_idxs_value = indexed.args[1:]
    is_inside = []
    for ivar, value in zip(ivars, ivar_idxs_value):
        if ivar.idx in value.atoms(Idx):
            is_inside.append(True)
            continue
        left_cond = _ensure_bool(value >= ivar.idx.lower, all_ivars)
        right_cond = _ensure_bool(value <= ivar.idx.upper, all_ivars)
        is_inside.append(bool(left_cond & right_cond))
    return is_inside


def _get_domain(indexed, ivars, all_ivars):
    ivar_idxs_value = indexed.args[1:]
    where = []
    for ivar, value in zip(ivars, ivar_idxs_value):
        if ivar.idx in value.atoms(Idx):
            where.append("bulk")
        elif _ensure_bool(value < ivar.idx.lower, all_ivars):
            where.append("left")
        elif _ensure_bool(value > ivar.idx.upper, all_ivars):
            where.append("right")
        else:
            where.append("bulk")
    return where


def _compute_bdc_on_ghost_node(indexed, ivars, bdcs, all_ivars):
    domains = _get_domain(indexed, ivars, all_ivars)
    ghost_node = indexed.args[1:]
    for ivar, _, domain in zip(ivars, ghost_node, domains):
        if domain == "left":
            yield bdcs[ivar][0].fdiff.subs(
                {ivar.idx: coord for ivar, coord in zip(ivars, ghost_node)}
            )
        if domain == "right":
            yield bdcs[ivar][1].fdiff.subs(
                {ivar.idx: coord for ivar, coord in zip(ivars, ghost_node)}
            )


def _include_bdc_in_localeq(all_unavailable_vars, all_available_bdcs, local_eq):
    all_idxs = set(
        [
            *chain(*[list(var.atoms(Idx)) for var in all_unavailable_vars]),
            *chain(*[list(bdc.atoms(Idx)) for bdc in all_available_bdcs]),
        ]
    )
    dummy_map = {idx: Dummy() for idx in all_idxs}
    reverse_dummy_map = {value: key for key, value in dummy_map.items()}
    all_unavailable_vars_ = set([var.subs(dummy_map) for var in all_unavailable_vars])
    all_available_bdcs_ = set([bdc.subs(dummy_map) for bdc in all_available_bdcs])
    all_available_bdcs_ = [
        bdc
        for bdc in all_available_bdcs_
        if _filter_relevent_equations(bdc, all_unavailable_vars_)
    ]
    logging.debug("ghost nodes: %s" % ", ".join(map(str, all_unavailable_vars_)))
    logging.debug(
        "using bdc to replace ghost nodes: %s"
        % ", ".join(map(str, all_available_bdcs_))
    )

    solved_variables = dict()
    for solved_ in list(
        solve(all_available_bdcs_, all_unavailable_vars_, dict=True, set=True)
    ):
        solved_variables.update(solved_)

    solved_variables = {
        key.subs(reverse_dummy_map): value.subs(reverse_dummy_map)
        for key, value in solved_variables.items()
    }

    logging.debug(
        "ghost nodes values: %s"
        % ", ".join(
            ["%s: %s" % (key, value) for key, value in solved_variables.items()]
        )
    )
    local_eq = local_eq.subs(solved_variables)
    return local_eq, solved_variables


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
        is_left = (coord < self.idx.lower).subs(self.N, 500)
        is_right = (coord > self.idx.upper).subs(self.N, 500)
        if not is_left and not is_right:
            return "bulk"
        if is_left:
            return "left"
        if is_right:
            return "right"

    def is_in_bulk(self, coord):
        return True if self.domain(coord) == "bulk" else False

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
        return all([ivar.is_in_bulk(coord) for ivar, coord in zip(self.ivars, coords)])

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
            deriv_product = product(
                self.independent_variables, fdiff.atoms(Derivative)
            )
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
                print(points)
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
            fdiff = fdiff.replace(
                ivar.symbol + a * ivar.step, ivar.idx + a
            )

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
        for dvar, pde in zip(self.dependent_variables, self):
            ivars = dvar.independent_variables
            eq = pde.fdiff
            domain = {}
            for i, ivar in enumerate(ivars, 1):
                lower_conds = [
                    (indexed.args[i] - ivar.idx.lower).subs(ivar.idx, ivar.symbol)
                    for indexed in eq.atoms(Indexed)
                    if ivar
                    in self.dependent_dict[str(indexed.args[0])].independent_variables
                ]
                upper_conds = [
                    (indexed.args[i] - ivar.idx.upper).subs(ivar.idx, ivar.symbol)
                    for indexed in eq.atoms(Indexed)
                    if ivar
                    in self.dependent_dict[str(indexed.args[0])].independent_variables
                ]
                cond = (
                    Ge(
                        ivar.idx,
                        max([solve(cond, ivar.symbol)[0] for cond in lower_conds]),
                    ),
                    Le(
                        ivar.idx,
                        min([solve(cond, ivar.symbol)[0] for cond in upper_conds]),
                    ),
                )

                domain[ivar] = cond
            self._domains.append(domain)
        self._unknown_nodes = []
        for domain in self._domains:
            unodes = {}
            for ivar, (lower_cond, upper_cond) in domain.items():
                if isinstance(lower_cond, (bool, BooleanTrue)):
                    left_unodes = []
                else:
                    left_unodes = np.arange(ivar.idx.lower, lower_cond.rhs).tolist()
                if isinstance(upper_cond, (bool, BooleanTrue)):
                    right_unodes = []
                else:
                    right_unodes = np.arange(
                        ivar.idx.upper, upper_cond.rhs, -1
                    ).tolist()
                unodes[ivar] = (left_unodes, right_unodes)
            self._unknown_nodes.append(unodes)

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
        for eq, sysdomain, dvar, unodes in zip(
            self, self._domains, self.dependent_variables, self._unknown_nodes
        ):
            system = {}
            ivars = dvar.independent_variables

            _unknown_nodes = unodes

            idxs = []
            for ivar, unode in _unknown_nodes.items():
                left_nodes, right_nodes = unode
                bulk_nodes = [ivar.idx]
                idx = chain(*[left_nodes, bulk_nodes, right_nodes])
                idxs.append(idx)
            coords_to_compute = list(product(*idxs))

            bdcs = self.boundary_conditions.values()

            for coords in coords_to_compute:
                logging.debug("evaluate coord: %s" % ", ".join(map(str, coords)))
                local_eq = eq.fdiff.subs(
                    {ivar.idx: coord for ivar, coord in zip(ivars, coords)}
                )
                subs_queue = Queue()
                unavailable_vars = [
                    indexed
                    for indexed in local_eq.atoms(Indexed)
                    if not all(_is_in_bulk(indexed, ivars, self.independent_variables))
                ]

                subs_queue.put(unavailable_vars)

                all_unavailable_vars = set()
                all_available_bdcs = set()
                solved_variables = dict()

                for i, unavailable_vars in enumerate(iter(subs_queue.get, [])):
                    available_bdcs = list(
                        chain(
                            *[
                                _compute_bdc_on_ghost_node(
                                    unavailable_var,
                                    bdc.keys(),
                                    bdc,
                                    self.independent_variables,
                                )
                                for unavailable_var, bdc in product(
                                    unavailable_vars, bdcs
                                )
                            ]
                        )
                    )
                    available_bdcs = [
                        bdc
                        for bdc in available_bdcs
                        if _filter_relevent_equations(bdc, unavailable_vars)
                    ]
                    if i > 5:
                        raise AssertionError
                    available_bdcs = set(
                        [bdc.subs(solved_variables) for bdc in available_bdcs]
                    )

                    all_available_bdcs = all_available_bdcs.union(set(available_bdcs))
                    all_unavailable_vars = all_unavailable_vars.union(
                        set(unavailable_vars)
                    )
                    indexed = set(
                        list(chain(*[bdc.atoms(Indexed) for bdc in all_available_bdcs]))
                    )
                    # indexed = local_eq.atoms(Indexed)
                    tosolve = [
                        idx
                        for idx in indexed
                        if not all(_is_in_bulk(idx, ivars, self.independent_variables))
                    ]
                    local_eq, solved_variables_ = _include_bdc_in_localeq(
                        tosolve, all_available_bdcs, local_eq
                    )
                    solved_variables.update(solved_variables_)

                    remaining_ghosts = [
                        indexed
                        for indexed in local_eq.atoms(Indexed)
                        if not all(
                            _is_in_bulk(indexed, ivars, self.independent_variables)
                        )
                    ]
                    if remaining_ghosts:
                        subs_queue.put(set(tosolve).union(set(remaining_ghosts)))
                    else:
                        subs_queue.put([])
                local_eq, solved_variables = _include_bdc_in_localeq(
                    all_unavailable_vars, all_available_bdcs, local_eq
                )

                remaining_ghosts = [
                    indexed
                    for indexed in local_eq.atoms(Indexed)
                    if not all(_is_in_bulk(indexed, ivars, self.independent_variables))
                ]
                if remaining_ghosts:
                    logging.error("remaining ghosts nodes! : %s" % remaining_ghosts)
                    raise RuntimeError(
                        "Solver not able to eliminate all ghost nodes."
                        " Check your boundary conditions !"
                    )

                logging.debug("local equation after subs: %s" % local_eq)
                logging.debug("Indexed in local eq: %s" % local_eq.atoms(Indexed))

                domain_map = []
                for coord, ivar in zip(coords, dvar.independent_variables):
                    if coord == ivar.idx:
                        domain_map.append(And(*sysdomain[ivar]))
                    else:
                        domain_map.append(Eq(ivar.idx, coord))
                domain_map = And(*domain_map)
                system[domain_map] = N(local_eq)
            self._system.append(list(system.items()))

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
        logging.info("get shapes...")
        self._get_shapes()
        logging.info("ensure bdc...")
        self._ensure_bdc()
        logging.info("build system...")
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
