#!/usr/bin/env python
# coding=utf8

import logging
import sys

# from .routines import F_Routine, J_Routine

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
      differential_equations : iterable of str or str
          the right hand sides of the partial differential equations written
          as :math:`\\frac{\partial U}{\partial t} = F(U)`, where the spatial
          derivative can be written as `dxxU` or `dx(U, 2)` or with the sympy
          notation `Derivative(U, x, x)`
      dependent_variables : iterable of str or str
          the dependent variables with the same order as the temporal
          derivative of the previous arg.
      parameters : iterable of str or str, optional, default None
          list of the parameters. Can be feed with a scalar of an array with
          the same size
      help_functions : None, optional
          All fields which have not to be solved with the time derivative but
          will be derived in space.

      Attributes
      ----------
      F : triflow.F_Routine
          Callable used to compute the right hand side of the dynamical system
      F_array : numpy.ndarray of sympy.Expr
          Symbolic expressions of the right hand side of the dynamical system
      J : triflow.J_Routine
          Callable used to compute the Jacobian of the dynamical system
      J_array : numpy.ndarray of sympy.Expr
          Symbolic expressions of the Jacobian side of the dynamical system

      Properties
      ----------
      fields_template:
          Model specific Fields container used to store andaccess to the model
          variables in an efficient way.

      Methods
      -------
      save: Save a binary of the Model with pre-optimized F and J routines

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
                 differential_equations,
                 dependent_variables,
                 parameters=None,
                 independent_variables=None,
                 bdc_conditions=None,
                 auxiliary_definitions=None,
                 compiler="theano",
                 hold_compilation=False):
        pass
