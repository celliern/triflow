#!/usr/bin/env python
# coding=utf-8

import logging
import re
import typing
from functools import partial, reduce
from itertools import chain, product
from operator import and_, mul
from queue import Queue

import attr
import numpy as np
from more_itertools import unique_everseen

from sympy import (
    Derivative,
    Eq,
    Expr,
    Function,
    Indexed,
    Max,
    Min,
    Symbol,
    Wild,
    solve,
    sympify,
)

from ..utils import solve_with_dummy
from .spatial_schemes import FiniteDifferenceScheme, chain_schemes
from .variables import DependentVariable as DVar
from .variables import IndependentVariable as IVar
from .variables import _convert_depvar_list, _convert_indepvar_list

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def _convert_pde_list(pdes):
    if isinstance(pdes, str):
        return [pdes]
    else:
        return pdes


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
        # for both side, left|right_cond can be true : in that case, no coords has to
        # be added : all that side is in the bulk.
        try:
            coords.extend(np.arange(ivar.idx.lower, left_cond.rhs))
        except AttributeError:
            pass
        try:
            coords.extend(
                np.arange(right_cond.rhs - ivar.N + 1, ivar.idx.upper - ivar.N + 1)
                + ivar.N
            )
        except AttributeError:
            pass
        available_coords[ivar] = coords
    return available_coords


def list_conditions(domain):
    available_conds = {}
    for ivar, (left_cond, right_cond) in domain.items():
        conds = []
        conds.append(left_cond & right_cond)
        # for both side, left|right_cond can be true : in that case, no coords has to
        # be added : all that side is in the bulk.
        try:
            conds.extend(
                [
                    Eq(ivar.idx, coord)
                    for coord in np.arange(ivar.idx.lower, left_cond.rhs)
                ]
            )
        except AttributeError:
            pass
        try:
            conds.extend(
                [
                    Eq(ivar.idx, coord)
                    for coord in np.arange(
                        right_cond.rhs - ivar.N + 1, ivar.idx.upper - ivar.N + 1
                    )
                    + ivar.N
                ]
            )
        except AttributeError:
            pass
        available_conds[ivar] = conds
    return available_conds


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
        lefts, rights = zip(*[bound.get(ivar, ivar.bound) for bound in bounds])
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


@attr.s
class PDEquation:
    equation = attr.ib(type=str)
    dependent_variables = attr.ib(
        type=typing.Tuple[DVar], converter=_convert_depvar_list
    )
    parameters = attr.ib(
        type=typing.Tuple[DVar], converter=_convert_depvar_list, default=[]
    )
    independent_variables = attr.ib(
        type=typing.Tuple[IVar],
        default=[],
        converter=_convert_indepvar_list,
        repr=False,
    )
    auxiliary_definitions = attr.ib(type=dict, default={})
    schemes = attr.ib(
        type=typing.Tuple[FiniteDifferenceScheme, ...],
        default=(FiniteDifferenceScheme(),),
    )
    symbolic_equation = attr.ib(init=False, repr=False)
    fdiff = attr.ib(init=False, repr=False)
    raw = attr.ib(type=bool, default=False, repr=False)

    # @scheme.validator
    # def check_scheme(self, attrs, scheme):
    #     return scheme in ["left", "right", "centered"]

    def __attrs_post_init__(self):
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

        fdiff = self.symbolic_equation
        fdiff = chain_schemes(self.schemes, fdiff)

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

            ivar = IVar(str(ivar))

            deriv_left = left_deriv(ivar, dvar)
            deriv_right = right_deriv(ivar, dvar)
            discretized_deriv = ap * deriv_left + am * deriv_right
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


@attr.s()
class Node:
    eq = attr.ib(type=Expr)
    dvar = attr.ib(type=DVar)
    coords = attr.ib(type=tuple)
    conds = attr.ib(type=Expr)
    mapper = attr.ib(type=dict)
    available_boundaries = attr.ib(type=dict)
    threshold = attr.ib(default=2, type=int)

    def __attrs_post_init__(self):
        logging.debug(
            "process node %s - %s (cond %s)" % (self.dvar, self.coords, self.conds)
        )
        self.subs = {}
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
        def sign_distance(distance, domain):
            if domain == "bulk":
                return 0
            if domain == "left":
                return -distance
            return distance

        def get_proper_boundary(boundaries, domains):
            for ivar, domain in zip(dvar.ivars, domains):
                left_bdc, right_bdc = boundaries[ivar]
                if domain == "left":
                    yield left_bdc.fdiff
                if domain == "right":
                    yield right_bdc.fdiff

        def select_coord(bdc, dvar, idx, domains):
            subs_coord = {
                ivar.idx: coord - sign_distance(1, domain)
                for ivar, coord, domain in zip(dvar.ivars, idx.args[1:], domains)
            }
            # try to guess if we have a dirichlet condition, in that case, evaluate
            # directly on the unknown coordinate.
            if idx not in bdc.subs(subs_coord).atoms(Indexed):
                subs_coord = {
                    ivar.idx: coord
                    for ivar, coord, domain in zip(dvar.ivars, idx.args[1:], domains)
                }
            return subs_coord

        for idx, domains, _ in self.outside_variables:
            logging.debug("evaluate ghost node %s %s" % (idx, domains))
            dvar = self.mapper[str(idx.args[0])]
            bdcs = list(get_proper_boundary(self.available_boundaries[dvar], domains))
            bdcs = [
                bdc.subs(select_coord(bdc, dvar, idx, domains)).subs(self.subs)
                for bdc in bdcs
            ]
            solved = solve_with_dummy(bdcs, idx)
            # If we have periodic condition, sometime, the solver cannot
            # choose between two way to obtain the solution :
            # U[-1, -1] can be solved by replacing the first coord then the
            # second, or the opposite. In that case, we arbitrary choose
            # one way.
            if not solved:
                solved = solve_with_dummy([bdcs[0]], idx)
            self.subs.update(solved)
        self.local_eq = self.local_eq.subs(self.subs)


@attr.s
class PDESys:
    evolution_equations = attr.ib(type=typing.List[str], converter=_convert_pde_list)
    dependent_variables = attr.ib(
        type=typing.List[DVar], converter=_convert_depvar_list
    )
    parameters = attr.ib(type=typing.List[DVar], converter=_convert_depvar_list)
    independent_variables = attr.ib(
        type=typing.List[IVar], default=[], converter=_convert_indepvar_list, repr=False
    )
    boundary_conditions = attr.ib(type=dict, default=None)
    auxiliary_definitions = attr.ib(type=dict, default=None)
    _domains = attr.ib(type=dict, default=None, init=False, repr=False)

    def _compute_domain(self):
        self._domains = []
        self._unknown_nodes = []
        for dvar, pde in zip(self.dependent_variables, self):
            bounds = [
                extract_bounds(indexed, self.mapper)
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
                            left_cond.subs(ivar.idx, ivar.idx),
                            dependent_variables=self.dependent_variables,
                            independent_variables=dvar.independent_variables,
                            raw=True,
                        ),
                        PDEquation(
                            right_cond.subs(ivar.idx, ivar.idx),
                            dependent_variables=self.dependent_variables,
                            independent_variables=dvar.independent_variables,
                            raw=True,
                        ),
                    )
                else:
                    def target_relevant_ivar(derivative, wrt):
                        return wrt == ivar.symbol

                    bdc[ivar] = [
                        PDEquation(
                            eq if eq else "d%s%s" % (ivar.name, dvar.name),
                            dependent_variables=self.dependent_variables,
                            independent_variables=dvar.independent_variables,
                            schemes=(
                                FiniteDifferenceScheme(
                                    scheme=scheme,
                                    offset=offset,
                                    pattern=target_relevant_ivar,
                                ),
                            ),
                        )
                        for scheme, offset, eq in zip(["right", "left"], [-1, 1], eqs)
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
                node = Node(
                    eq=pde.fdiff,
                    dvar=dvar,
                    coords=coords,
                    conds=conds,
                    mapper=self.mapper,
                    available_boundaries=self.boundary_conditions,
                )
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
        logging.info("get shapes...")
        self._get_shapes()
        logging.info("build system...")
        self._build_system()
        logging.info("done")

    @property
    def dependent_dict(self):
        return {depvar.name: depvar for depvar in self.dependent_variables}

    @property
    def independent_dict(self):
        return {
            ivar.name: ivar
            for ivar in set(chain(*[dvar.ivars for dvar in self.dependent_variables]))
        }

    @property
    def parameters_dict(self):
        return {par.name: par for par in self.parameters}

    @property
    def mapper(self):
        return dict(
            **self.dependent_dict, **self.parameters_dict, **self.independent_dict
        )

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
