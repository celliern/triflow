#!/usr/bin/env python
# coding=utf8

"""This module regroups all the implemented temporal schemes.
They are written as callable class which take the model and some control
arguments at the init, and perform a computation step every time they are
called.

The following solvers are implemented:
    * Backward and Forward Euler, Crank-Nicolson method (with the Theta class)
    * Some Rosenbrock Wanner schemes (up to the 6th order) with time controler
    * All the scipy.integrate.ode integrators with the scipy_ode class.
"""

import logging

import numpy as np
import scipy.sparse as sps
from scipy.integrate import ode
from scipy.linalg import norm
from toolz import memoize

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class ROW_general:
    """Rosenbrock Wanner class of temporal solvers

    The implementation and the different parameters can be found in
    http://www.digibib.tu-bs.de/?docid=00055262
    """

    @memoize
    def __cache__(self, N):
        Id = sps.eye(N, format='csc')
        return Id

    def __init__(self, model, alpha, gamma, b, b_pred=None,
                 time_stepping=False, tol=None, max_iter=None, dt_min=None):
        self._internal_dt = None
        self._model = model
        self._alpha = alpha
        self._gamma = gamma
        self._b = b
        self._b_pred = b_pred
        self._s = len(b)
        self._time_control = time_stepping
        self._tol = tol
        self._max_iter = max_iter
        self._dt_min = dt_min

    def __call__(self, t, fields, dt, pars,
                 hook=lambda t, fields, pars: (fields, pars)):
        """Perform a step of the solver: took a time and a system state as a
          triflow Fields container and return the next time step with updated
          container.

          Parameters
          ----------
          t : float
              actual time step
          fields : triflow.Fields
              actual system state in a triflow Fields
          dt : float
              temporal step-size
          pars : dict
              physical parameters of the model
          hook : callable, optional
              any callable taking the actual time, fields and parameters and return modified fields and parameters. Will be called every internal time step and can be used to include time dependent or conditionnal parameters, boundary conditions...
          container

          Returns
          -------
          tuple : t, fields
              updated time and fields container

          Raises
          ------
          NotImplementedError
              raised if a time stepping is requested but the scheme do not provide the b predictor coefficients.
          ValueError
              raised if time_stepping is True and tol is not provided.
          """  # noqa
        if self._time_control:
            return self._variable_step(t, fields, dt, pars,
                                       hook=hook)

        t, fields, _ = self._fixed_step(t, fields, dt, pars,
                                        hook=hook)
        fields, pars = hook(t, fields, pars)
        return t, fields

    def _fixed_step(self, t, fields, dt, pars,
                    hook=lambda t, fields, pars: (fields, pars)):
        fields = fields.copy()
        fields, pars = hook(t, fields, pars)
        J = self._model.J(fields, pars)
        Id = self.__cache__(fields.uflat.size)
        A = Id - self._gamma[0, 0] * dt * J
        luf = sps.linalg.splu(A)
        ks = []
        fields_i = fields.copy()
        for i in np.arange(self._s):
            fields_i.fill(fields.uflat +
                          sum([self._alpha[i, j] * ks[j]
                               for j in range(i)]))
            F = self._model.F(fields_i, pars)
            ks.append(luf.solve(dt * F + dt * (J @ sum([self._gamma[i, j] *
                                                        ks[j]
                                                        for j
                                                        in range(i)])
                                               if i > 0 else 0)))
        U = fields.uflat.copy()
        U = U + sum([bi * ki for bi, ki in zip(self._b, ks)])

        U_pred = (U + sum([bi * ki
                           for bi, ki
                           in zip(self._b_pred, ks)])
                  if self._b_pred is not None else None)
        fields.fill(U)

        return t + dt, fields, (norm(U - U_pred, np.inf)
                                if U_pred is not None else None)

    def _variable_step(self, t, fields, dt, pars,
                       hook=lambda t, fields, pars: (fields, pars)):

        self._next_time_step = t + dt
        self._internal_iter = 0
        dt = self._internal_dt = (1E-6 if self._internal_dt is None
                                  else self._internal_dt)
        while True:
            self._err = None
            while (self._err is None or self._err > self._tol):
                _, newfields, self._err = self._fixed_step(t,
                                                           fields,
                                                           dt,
                                                           pars,
                                                           hook)
                logging.debug(f"error: {self._err}")
                dt = (0.9 * dt * np.sqrt(self._tol / self._err))
            fields = newfields.copy()
            logging.debug(f'dt computed after err below tol: {dt}')
            logging.debug(f'ROS_vart, t {t}')
            if t + dt >= self._next_time_step:
                logging.debug('new t more than next expected time step')
                dt = self._next_time_step - t
                logging.debug(f'dt falling to {dt}')
            self._internal_dt = dt
            t = t + self._internal_dt
            self._internal_iter += 1
            if np.isclose(t, self._next_time_step):
                self._internal_dt = dt
                fields, pars = hook(t, fields, pars)
                return self._next_time_step, fields
            if self._internal_iter > (self._max_iter
                                      if self._max_iter
                                      else self._internal_iter + 1):
                raise RuntimeError("Rosebrock internal iteration "
                                   "above max iterations authorized")
            if self._internal_dt < (self._dt_min
                                    if self._dt_min
                                    else self._internal_dt * .5):
                raise RuntimeError("Rosebrock internal time step "
                                   "less than authorized")


class ROS2(ROW_general):
    """Second order Rosenbrock scheme, without time stepping

    Parameters
    ----------
    model : triflow.Model
        triflow Model
    """

    def __init__(self, model):
        gamma = np.array([[2.928932188134E-1, 0],
                          [-5.857864376269E-1, 2.928932188134E-1]])
        alpha = np.array([[0, 0],
                          [1, 0]])
        b = np.array([1 / 2, 1 / 2])
        super().__init__(model, alpha, gamma, b, time_stepping=False)


class ROS3PRw(ROW_general):
    """Third order Rosenbrock scheme, with time stepping

      Parameters
      ----------
      model : triflow.Model
          triflow Model
      tol : float, optional, default 1E-2
          tolerance factor for the time stepping. The time step will adapt to ensure that the maximum relative error on all fields stay under that value.
      time_stepping : bool, optional, default True
          allow a variable internal time-step to ensure good agreement between computing performance and accuracy.
      max_iter : float or None, optional, default None
          maximum internal iteration allowed
      dt_min : float or None, optional, default None
          minimum internal time step allowed
      """  # noqa

    def __init__(self, model, tol=1E-2, time_stepping=True,
                 max_iter=None, dt_min=None):
        alpha = np.zeros((3, 3))
        gamma = np.zeros((3, 3))
        gamma_i = 7.8867513459481287e-01
        b = [5.0544867840851759e-01,
             -1.1571687603637559e-01,
             6.1026819762785800e-01]
        b_pred = [2.8973180237214197e-01,
                  1.0000000000000001e-01,
                  6.1026819762785800e-01]

        alpha[1, 0] = 2.3660254037844388e+00
        alpha[2, 0] = 5.0000000000000000e-01
        alpha[2, 1] = 7.6794919243112270e-01
        gamma[0, 0] = gamma[1, 1] = gamma[2, 2] = gamma_i
        gamma[1, 0] = -2.3660254037844388e+00
        gamma[2, 0] = -8.6791218280355165e-01
        gamma[2, 1] = -8.7306695894642317e-01
        super().__init__(model, alpha, gamma, b, b_pred=b_pred,
                         time_stepping=time_stepping, tol=tol,
                         max_iter=max_iter, dt_min=dt_min)


class ROS3PRL(ROW_general):
    """4th order Rosenbrock scheme, with time stepping

      Parameters
      ----------
      model : triflow.Model
          triflow Model
      tol : float, optional, default 1E-2
          tolerance factor for the time stepping. The time step will adapt to ensure that the maximum relative error on all fields stay under that value.
      time_stepping : bool, optional, default True
          allow a variable internal time-step to ensure good agreement between computing performance and accuracy.
      max_iter : float or None, optional, default None
          maximum internal iteration allowed
      dt_min : float or None, optional, default None
          minimum internal time step allowed
      """  # noqa

    def __init__(self, model, tol=1E-2, time_stepping=True,
                 max_iter=None, dt_min=None):
        alpha = np.zeros((4, 4))
        gamma = np.zeros((4, 4))
        gamma_i = 4.3586652150845900e-01

        b = [2.1103008548132443e-03,
             8.8607515441580453e-01,
             -3.2405197677907682e-01,
             4.3586652150845900e-01]
        b_pred = [5.0000000000000000e-01,
                  3.8752422953298199e-01,
                  -2.0949226315045236e-01,
                  3.2196803361747034e-01]
        alpha[1, 0] = .5
        alpha[2, 0] = .5
        alpha[2, 1] = .5
        alpha[3, 0] = .5
        alpha[3, 1] = .5
        alpha[3, 2] = 0
        for i in range(len(b)):
            gamma[i, i] = gamma_i
        gamma[1, 0] = -5.0000000000000000e-01
        gamma[2, 0] = -7.9156480420464204e-01
        gamma[2, 1] = 3.5244216792751432e-01
        gamma[3, 0] = -4.9788969914518677e-01
        gamma[3, 1] = 3.8607515441580453e-01
        gamma[3, 2] = -3.2405197677907682e-01
        super().__init__(model, alpha, gamma, b, b_pred=b_pred,
                         time_stepping=time_stepping, tol=tol,
                         max_iter=max_iter, dt_min=dt_min)


class RODASPR(ROW_general):
    """6th order Rosenbrock scheme, with time stepping

      Parameters
      ----------
      model : triflow.Model
          triflow Model
      tol : float, optional, default 1E-2
          tolerance factor for the time stepping. The time step will adapt to ensure that the maximum relative error on all fields stay under that value.
      time_stepping : bool, optional, default True
          allow a variable internal time-step to ensure good agreement between computing performance and accuracy.
      max_iter : float or None, optional, default None
          maximum internal iteration allowed
      dt_min : float or None, optional, default None
          minimum internal time step allowed
      """  # noqa

    def __init__(self, model, tol=1E-2, time_stepping=True,
                 max_iter=None, dt_min=None):
        alpha = np.zeros((6, 6))
        gamma = np.zeros((6, 6))
        b = [-7.9683251690137014E-1,
             6.2136401428192344E-2,
             1.1198553514719862E00,
             4.7198362114404874e-1,
             -1.0714285714285714E-1,
             2.5e-1]
        b_pred = [-7.3844531665375115e0,
                  -3.0593419030174646e-1,
                  7.8622074209377981e0,
                  5.7817993590145966e-1,
                  2.5e-1,
                  0]
        alpha[1, 0] = 7.5E-1
        alpha[2, 0] = 7.5162877593868457E-2
        alpha[2, 1] = 2.4837122406131545E-2
        alpha[3, 0] = 1.6532708886396510e0
        alpha[3, 1] = 2.1545706385445562e-1
        alpha[3, 2] = -1.3157488872766792e0
        alpha[4, 0] = 1.9385003738039885e1
        alpha[4, 1] = 1.2007117225835324e0
        alpha[4, 2] = -1.9337924059522791e1
        alpha[4, 3] = -2.4779140110062559e-1
        alpha[5, 0] = -7.3844531665375115e0
        alpha[5, 1] = -3.0593419030174646e-1
        alpha[5, 2] = 7.8622074209377981e0
        alpha[5, 3] = 5.7817993590145966e-1
        alpha[5, 4] = 2.5e-1
        gamma_i = .25
        for i in range(len(b)):
            gamma[i, i] = gamma_i
        gamma[1, 0] = -7.5e-1
        gamma[2, 0] = -8.8644e-2
        gamma[2, 1] = -2.868897e-2
        gamma[3, 0] = -4.84700e0
        gamma[3, 1] = -3.1583e-1
        gamma[3, 2] = 4.9536568e0
        gamma[4, 0] = -2.67694569e1
        gamma[4, 1] = -1.5066459e0
        gamma[4, 2] = 2.720013e1
        gamma[4, 3] = 8.25971337e-1
        gamma[5, 0] = 6.58762e0
        gamma[5, 1] = 3.6807059e-1
        gamma[5, 2] = -6.74235e0
        gamma[5, 3] = -1.061963e-1
        gamma[5, 4] = -3.57142857e-1
        super().__init__(model, alpha, gamma, b, b_pred=b_pred,
                         time_stepping=time_stepping, tol=tol,
                         max_iter=max_iter, dt_min=dt_min)


class scipy_ode:
    """Proxy written around the scipy.integrate.ode class. Give access to all
      the scpy integrators.

      Parameters
      ----------
      model : triflow.Model
          triflow Model
      integrator : str, optional, default 'vode'
          name of the chosen scipy integration scheme.
      **integrator_kwargs
          extra arguments provided to the scipy integration scheme.
      """  # noqa

    def __init__(self, model, integrator='vode', **integrator_kwargs):
        def func_scipy_proxy(t, U, fields, pars, hook):
            fields.fill(U)
            fields, pars = hook(t, fields, pars)
            return model.F(fields, pars)

        def jacob_scipy_proxy(t, U, fields, pars, hook):
            fields.fill(U)
            fields, pars = hook(t, fields, pars)
            return model.J(fields, pars, sparse=False)

        self._solv = ode(func_scipy_proxy,
                         jac=jacob_scipy_proxy)
        self._solv.set_integrator(integrator, **integrator_kwargs)

    def __call__(self, t, fields, dt, pars,
                 hook=lambda t, fields, pars: (fields, pars)):
        """Perform a step of the solver: took a time and a system state as a
          triflow Fields container and return the next time step with updated
          container.

          Parameters
          ----------
          t : float
              actual time step
          fields : triflow.Fields
              actual system state in a triflow Fields
          dt : float
              temporal step-size
          pars : dict
              physical parameters of the model
          hook : callable, optional
              any callable taking the actual time, fields and parameters and return modified fields and parameters. Will be called every internal time step and can be used to include time dependent or conditionnal parameters, boundary conditions...
          container

          Returns
          -------
          tuple : t, fields
              updated time and fields container

          Raises
          ------
          RuntimeError
              Description
          """  # noqa

        solv = self._solv
        fields, pars = hook(t, fields, pars)
        solv.set_initial_value(fields.uflat, t)
        solv.set_f_params(fields, pars, hook)
        solv.set_jac_params(fields, pars, hook)
        U = solv.integrate(t + dt)
        fields.fill(U)
        fields, _ = hook(t + dt, fields, pars)
        return t + dt, fields


class Theta:
    """Simple theta-based scheme where theta is a weight
          if theta = 0, the scheme is a forward-euler scheme
          if theta = 1, the scheme is a backward-euler scheme
          if theta = 0.5, the scheme is called a Crank-Nicolson scheme

      Parameters
      ----------
      model : triflow.Model
          triflow Model
      theta : int, optional, default 1
          weight of the theta-scheme
      solver : callable, optional, default scipy.sparse.linalg.spsolve
          method able to solve a Ax = b linear equation with A a sparse matrix. Take A and b as argument and return x.
      """  # noqa

    def __init__(self, model, theta=1, solver=sps.linalg.spsolve):
        self._model = model
        self._theta = theta
        self._solver = solver

    def __call__(self, t, fields, dt, pars,
                 hook=lambda t, fields, pars: (fields, pars)):
        """Perform a step of the solver: took a time and a system state as a
          triflow Fields container and return the next time step with updated
          container.

          Parameters
          ----------
          t : float
              actual time step
          fields : triflow.Fields
              actual system state in a triflow Fields container
          dt : float
              temporal step-size
          pars : dict
              physical parameters of the model
          hook : callable, optional
              any callable taking the actual time, fields and parameters and return modified fields and parameters. Will be called every internal time step and can be used to include time dependent or conditionnal parameters, boundary conditions...

          Returns
          -------
          tuple : t, fields
              updated time and fields container
          """  # noqa

        fields = fields.copy()
        fields, pars = hook(t, fields, pars)
        F = self._model.F(fields, pars)
        J = self._model.J(fields, pars)
        U = fields.uflat
        B = dt * (F - self._theta * J @ U) + U
        J = (sps.identity(U.size,
                          format='csc') -
             self._theta * dt * J)
        fields.fill(self._solver(J, B))
        fields, _ = hook(t + dt, fields, pars)
        return t + dt, fields
