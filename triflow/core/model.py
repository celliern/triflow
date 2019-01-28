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
      evolution_equations : Union[Sequence[str], str]
          the right hand sides of the partial differential equations written
          as :math:`\\frac{\\partial U}{\\partial t} = F(U)`, where the spatial
          derivative can be written as `dxxU` or `dx(U, 2)` or with the sympy
          notation `Derivative(U, x, x)`
      dependent_variables : Union[Sequence[str], str]
          the dependent variables with the same order as the temporal
          derivative of the previous arg. 
      parameters : Optional[Union[Sequence[str], str]]
          list of the parameters. Can be feed with a scalar of an array with
          the same size. They can be derivated in space as well.
      boundary_conditions : Optional[Union[str, Dict[str, Dict[str, Tuple[str, str]]]]]
           Can be either "noflux" (default behavior), "periodic", or a dictionary that
           follow this structure :
           {dependent_var: {indep_var: (left_boundary, right_boundary)}}, the boundaries
           being written as residual (rhs - lhs = 0).
           For example, an hybrid Dirichlet / Neumann flux will be written as
           {"U": {"x": (dxU - 2, U - 3)}}. If a boundary is None, a nul flux will be
           applied.
      auxiliary_definitions : Optional[Dict[str, str]]
           Substitution dictionnary, useful to write complexe systems. The key of this
           dictionnary will be substitued everywhere in the evolution equations as well
           as in boundary conditions.
      compiler: str, default to "numpy"
           A registered compiler that will take the discretized system and expose the evolution
           equations routine ``F`` as well as an optional Jacobian routine ``J``. If the later is
           not provided by the compiler, the implicit methods will not be available.
           TODO: make an efficient jacobian approximation for that case, given the bandwidth.
           For now, there is the ``numpy`` compiler, that provide efficient (but not optimal)
           computation with minimal dependencies, and ``numba`` compiler that provide a LLVM based
           routine that allow faster parallel computation at the costs of a (sometime long) warm-up.
           The numba one is less tested and stable, but provide a potential huge speed-up for explicit
           methods.
      compiler_kwargs: Optional[Dict] : supplementary arguments provided to the compiler.

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

      A 2D pure advection model, without / with upwind scheme.

      >>> from triflow import Model
      >>> model = Model("c_x * dxU + c_y * dyU", "U(x, y)",
      ...               ["c_x", "c_y"])
      >>> model = Model("upwind(c_x, U, x, 2) + upwind(c_y, U, y, 2)",
      ...               "U(x, y)",
      ...               ["c_x", "c_y"])


      A 2D diffusion model, with hybrid boundary conditions (Dirichlet, Neuman, Robin and No-flux).

      >>> from triflow import Model
      >>> model = Model("dxxU + dyyU", "U(x, y)",
      ...               boundary_conditions={"U": {"x": ("U - 3", "dxU + 5")
      ...                                          "y": ("dyU - (U - 3)", None)}
      ...                                          })

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
