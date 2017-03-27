#!/usr/bin/env python
# coding=utf8

import logging

import scipy.sparse as sps
from scipy.linalg import norm
from scipy.integrate import ode
from toolz import memoize
import numpy as np


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class ROW_general:
    """http://www.digibib.tu-bs.de/?docid=00055262"""

    @memoize
    def __cache__(self, N):
        Id = sps.eye(N, format='csc')
        return Id

    def __init__(self, model, alpha, gamma, b, b_pred=None):
        self.internal_dt = None
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.b = b
        self.b_pred = b_pred
        self.s = len(b)

    def __call__(self, fields, t, dt, pars,
                 hook=lambda fields, t, pars: (fields, pars)):
        if pars.get('time_stepping', False):
            try:
                if self.b_pred is None:
                    raise NotImplementedError(
                        'No b predictor provided for '
                        'this scheme, unable to use '
                        'variable time stepping, falling '
                        'to constant time step = %f' % dt
                    )
                return self.variable_step(fields, t, dt, pars,
                                          hook=hook)
            except NotImplementedError as e:
                logging.warning(e)

        fields, t, _ = self.fixed_step(fields, t, dt, pars,
                                       hook=hook)
        fields, pars = hook(fields, t, pars)
        return fields, t

    def fixed_step(self, fields, t, dt, pars,
                   hook=lambda fields, t, pars: (fields, pars)):
        fields = fields.copy()
        fields, pars = hook(fields, t, pars)
        J = self.model.J(fields, pars)
        Id = self.__cache__(fields.uflat.size)
        A = Id - self.gamma[0, 0] * dt * J
        luf = sps.linalg.splu(A)
        ks = []
        fields_i = fields.copy()
        for i in np.arange(self.s):
            fields_i.fill(fields.uflat +
                          sum([self.alpha[i, j] * ks[j]
                               for j in range(i)]))
            F = self.model.F(fields_i, pars)
            ks.append(luf.solve(dt * F + dt * (J @ sum([self.gamma[i, j] *
                                                        ks[j]
                                                        for j
                                                        in range(i)])
                                               if i > 0 else 0)))
        U = fields.uflat.copy()
        U = U + sum([bi * ki for bi, ki in zip(self.b, ks)])

        U_pred = (U + sum([bi * ki
                           for bi, ki
                           in zip(self.b_pred, ks)])
                  if self.b_pred is not None else None)
        fields.fill(U)

        return fields, t + dt, (norm(U - U_pred, np.inf)
                                if U_pred is not None else None)

    def variable_step(self, fields, t, dt, pars,
                      hook=lambda fields, t, pars: (fields, pars)):

        self.next_time_step = t + dt
        self.internal_iter = 0
        dt = self.internal_dt = (1E-6 if self.internal_dt is None
                                 else self.internal_dt)
        while True:
            self.err = None
            while (self.err is None or self.err > pars['tol']):
                newfields, _, self.err = self.fixed_step(fields,
                                                         t,
                                                         dt,
                                                         pars,
                                                         hook)
                logging.debug(f"error: {self.err}")
                dt = (0.9 * dt * np.sqrt(pars['tol'] / self.err))
            fields = newfields.copy()
            logging.debug(f'dt computed after err below tol: {dt}')
            logging.debug(f'ROS_vart, t {t}')
            if t + dt >= self.next_time_step:
                logging.debug('new t more than next expected time step')
                dt = self.next_time_step - t
                logging.debug(f'dt falling to {dt}')
            self.internal_dt = dt
            t = t + self.internal_dt
            self.internal_iter += 1
            if np.isclose(t, self.next_time_step):
                self.internal_dt = dt
                fields, pars = hook(fields, t, pars)
                return fields, self.next_time_step
            if self.internal_iter > pars.get('max_iter',
                                             self.internal_iter + 1):
                raise RuntimeError("Rosebrock internal iteration "
                                   "above max iterations authorized")
            if self.internal_dt < pars.get('dt_min', self.internal_dt * .5):
                raise RuntimeError("Rosebrock internal time step "
                                   "less than authorized")


class ROS2(ROW_general):
    """Second order Rosenbrock scheme, without time stepping"""

    def __init__(self, model):
        gamma = np.array([[2.928932188134E-1, 0],
                          [-5.857864376269E-1, 2.928932188134E-1]])
        alpha = np.array([[0, 0],
                          [1, 0]])
        b = np.array([1 / 2, 1 / 2])
        super().__init__(model, alpha, gamma, b)


class ROS3PRw(ROW_general):
    """docstring for FE"""

    def __init__(self, model):
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
        super().__init__(model, alpha, gamma, b, b_pred=b_pred)


class ROS3PRL(ROW_general):
    """docstring for FE"""

    def __init__(self, model):
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
        super().__init__(model, alpha, gamma, b, b_pred=b_pred)


class RODASPR(ROW_general):
    """docstring for FE"""

    def __init__(self, model):
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
        super().__init__(model, alpha, gamma, b, b_pred=b_pred)


class scipy_ode:
    """docstring for FE"""

    def __init__(self, model, integrator='vode', **integrator_kwargs):
        def func_scipy_proxy(t, U, fields, pars, hook):
            fields.fill(U)
            fields, pars = hook(fields, t, pars)
            return model.F(fields, pars)

        def jacob_scipy_proxy(t, U, fields, pars, hook):
            fields.fill(U)
            fields, pars = hook(fields, t, pars)
            return model.J(fields, pars, sparse=False)

        self.solv = ode(func_scipy_proxy,
                        jac=jacob_scipy_proxy)
        self.solv.set_integrator(integrator, **integrator_kwargs)

    def __call__(self, fields, t, dt, pars,
                 hook=lambda fields, t, pars: (fields, pars)):
        solv = self.solv
        fields, pars = hook(fields, t, pars)
        solv.set_initial_value(fields.uflat, t)
        solv.set_f_params(fields, pars, hook)
        solv.set_jac_params(fields, pars, hook)
        U = solv.integrate(t + dt)
        fields.fill(U)
        if solv.successful:
            fields, _ = hook(fields, t + dt, pars)
            return fields, t + dt
        else:
            raise RuntimeError


class Theta:
    """docstring for FE_scheme"""

    def __init__(self, model, theta=1):
        self.model = model
        self.theta = theta

    def __call__(self, fields, t, dt, pars,
                 hook=lambda fields, t, pars: (fields, pars),
                 solver=sps.linalg.spsolve):
        fields = fields.copy()
        fields, pars = hook(fields, t, pars)
        F = self.model.F(fields, pars)
        J = self.model.J(fields, pars)
        U = fields.uflat
        B = dt * (F - self.theta * J @ U) + U
        J = (sps.identity(U.size,
                          format='csc') -
             self.theta * dt * J)
        fields.fill(solver(J, B))
        fields, _ = hook(fields, t + dt, pars)
        return fields, t + dt
