#!/usr/bin/env python
# coding=utf-8

import logging
import sys
from itertools import chain
import xarray as xr

import forge

from .compilers import get_compiler
from .system import PDESys

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

sys.setrecursionlimit(40000)
EPS = 1E-6


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
          as :math:`\\frac{\partial U}{\partial t} = F(U)`, where the spatial
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

    def __init__(self,
                 evolution_equations,
                 dependent_variables,
                 parameters=[],
                 independent_variables=[],
                 boundary_conditions={},
                 auxiliary_definitions={},
                 compiler="theano"):

        self.pdesys = PDESys(
            evolution_equations=evolution_equations,
            dependent_variables=dependent_variables,
            independent_variables=independent_variables,
            parameters=parameters,
            boundary_conditions=boundary_conditions,
            auxiliary_definitions=auxiliary_definitions)
        self.compiler = get_compiler(compiler)(self.pdesys)

        self.F = self.compiler.F
        self.J = self.compiler.J
        self.U_from_fields = self.compiler.U_from_fields
        self.fields_from_U = self.compiler.fields_from_U

        dvar_names = [dvar.name for dvar in self.pdesys.dependent_variables]
        ivar_names = [ivar.name for ivar in self.pdesys.independent_variables]
        par_names = [par.name for par in self.pdesys.parameters]

        @forge.sign(*[
            forge.arg(name)
            for name in chain(dvar_names, ivar_names, par_names)
        ])
        def create_dataset(**kwargs):
            dvars = {
                dvar.name:
                ([ivar.name for ivar in dvar.independent_variables],
                 kwargs[dvar.name])
                for dvar in chain(
                    self.pdesys.dependent_variables,
                    self.pdesys.parameters,
                )
            }
            coords = {
                ivar.name: kwargs[ivar.name]
                for ivar in self.pdesys.independent_variables
            }

            return xr.Dataset(data_vars=dvars, coords=coords)

        self.fields_template = create_dataset
