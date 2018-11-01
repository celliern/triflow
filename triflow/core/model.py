#!/usr/bin/env python
# coding=utf-8

import logging
import sys
from copy import deepcopy
import warnings
from itertools import chain

from xarray import Dataset

from .compilers import get_compiler
from .grid_builder import GridBuilder
from .system import PDESys

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

sys.setrecursionlimit(40000)
EPS = 1e-6


class Model:
    """Contain finite difference approximation and routine of the dynamicalsystem

      Take a mathematical form as input, use Sympy to transform it as a symbolic
      expression, perform the finite difference approximation and expose theano
      optimized routine for both the right hand side of the dynamical system and
      Jacobian matrix approximation.

      Parameters
      ----------
      evolution_equations : iterable of str or str
          the right hand sides of the partial differential equations written
          as :math:`\\frac{\\partial U}{\\partial t} = F(U)`, where the spatial
          derivative can be written as `dxxU` or `dx(U, 2)` or with the sympy
          notation `Derivative(U, x, x)`
      dependent_variables : iterable of str or str
          the dependent variables with the same order as the temporal
          derivative of the previous arg.
      parameters : iterable of str or str, optional, default None
          list of the parameters. Can be feed with a scalar of an array with
          the same size. They can be derivated in space as well.

      Methods
      -------
      F : Compute the right hand side of the dynamical system
      J : Compute the Jacobian of the dynamical system as sparce csc matrix

      Examples
      --------
      A simple diffusion equation:

      >>> from triflow import Model
      >>> model = Model("k * dxxU", "U", "k")

      A coupled system of convection-diffusion equation:

      >>> from triflow import Model
      >>> model = Model(["k1 * dxxU - c1 * dxV",
      ...                "k2 * dxxV - c2 * dxU",],
      ...                ["U", "V"], ["k1", "k2", "c1", "c2"])
      """  # noqa

    def __init__(
        self,
        evolution_equations,
        dependent_variables,
        parameters=None,
        independent_variables=None,
        boundary_conditions=None,
        auxiliary_definitions=None,
        compiler="numpy",
        compiler_kwargs=None,
    ):
        if parameters is None:
            parameters = []
        if independent_variables is None:
            independent_variables = []
        if boundary_conditions is None:
            boundary_conditions = {}
        if auxiliary_definitions is None:
            auxiliary_definitions = {}
        if compiler_kwargs is None:
            compiler_kwargs = {}

        parameters = deepcopy(parameters)
        independent_variables = deepcopy(independent_variables)
        boundary_conditions = deepcopy(boundary_conditions)
        auxiliary_definitions = deepcopy(auxiliary_definitions)

        self.pdesys = PDESys(
            evolution_equations=evolution_equations,
            dependent_variables=dependent_variables,
            independent_variables=independent_variables,
            parameters=parameters,
            boundary_conditions=boundary_conditions,
            auxiliary_definitions=auxiliary_definitions,
        )
        self.grid_builder = GridBuilder(self.pdesys)

        self.compiler = get_compiler(compiler)(
            self.pdesys, self.grid_builder, **compiler_kwargs
        )

        self.F = self.compiler.F
        try:
            self.J = self.compiler.J
        except AttributeError:
            warnings.warn(
                "current compiler %s lack of routine to compute the jacobian matrix. "
                "Only explicite scheme will be working." % self.compiler.name
            )
        self.U_from_fields = self.grid_builder.U_from_fields
        self.fields_from_U = self.grid_builder.fields_from_U

        dvar_names = [dvar.name for dvar in self.pdesys.dependent_variables]
        ivar_names = [ivar.name for ivar in self.pdesys.independent_variables]
        par_names = [par.name for par in self.pdesys.parameters]

        _create_dataset_template = """
def create_dataset(*, {args}):
    \"""Create a xarray Dataset as expected by the model input.\"""
    dvars = {dvars}
    coords = {coords}

    return Dataset(data_vars=dvars, coords=coords)
self.Fields = self.fields_template = create_dataset
        """.format(
            args=", ".join([*ivar_names, *dvar_names, *par_names]),
            coords="dict(%s)"
            % ", ".join(["%s=%s" % (name, name) for name in ivar_names]),
            dvars="dict(%s)"
            % ", ".join(
                [
                    "{dvar_name}=({coords}, {dvar_name})".format(
                        dvar_name=dvar.name,
                        coords=["%s" % ivar for ivar in dvar.independent_variables],
                    )
                    for dvar in chain(
                        self.pdesys.dependent_variables, self.pdesys.parameters
                    )
                ]
            ),
        )

        exec(_create_dataset_template, dict(self=self, chain=chain, Dataset=Dataset))
