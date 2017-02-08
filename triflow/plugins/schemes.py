#!/usr/bin/env python
# coding=utf8

import logging

import scipy.sparse as sps
from scipy.linalg import norm
from scipy.integrate import ode
from collections import deque
import numpy as np


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class FE_scheme:
    """docstring for FE_scheme"""

    def __init__(self, model):
        self.solv = ode(lambda t, u, args: model.F(u, args))
        self.solv.set_integrator('dopri5')

    def __call__(self, U, t, dt, pars, hook=lambda x: x):
        solv = self.solv
        solv.set_initial_value(U, t)
        solv.set_f_params(pars)
        U = solv.integrate(t + dt)
        if solv.successful:
            U = hook(U)
            return U, t + dt
        else:
            raise RuntimeError


class BDF2_scheme:
    """docstring for FE_scheme"""

    def __init__(self, model):
        self.model = model
        self.Uhist = deque([], 2)

    def __call__(self, U, t, dt, pars, hook=lambda x: x):
        N = int(U.size / self.model.nvar)
        Id = sps.identity(self.model.nvar * N, format='csc')
        if len(self.Uhist) <= 2:
            F = self.model.F(U, pars)
            J = self.model.J(U, pars)
            B = dt * (F - J @ U) + U
            J = (Id - dt * J)
            U = sps.linalg.lgmres(J, B, x0=U)[0]
            self.Uhist.append(U.copy())
            return U, t + dt
        Un = self.Uhist[-1]
        Unm1 = self.Uhist[-2]
        F = self.model.F(U, pars)
        J = self.model.J(U, pars)
        B = ((4 / 3 * Id - 2 / 3 * dt * J) @ Un -
             1 / 3 * Unm1 +
             2 / 3 * dt * F)
        J = (Id -
             2 / 3 * dt * J)
        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = hook(U)
        self.Uhist.append(U.copy())
        return U, t + dt


class theta_scheme:
    """docstring for FE_scheme"""

    def __init__(self, model):
        self.model = model

    def __call__(self, U, t, dt, pars, hook=lambda x: x):
        F = self.model.F(U, pars)
        J = self.model.J(U, pars)
        B = dt * (F - pars['theta'] * J @ U) + U
        J = (sps.identity(U.size,
                          format='csc') -
             pars['theta'] * dt * J)
        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = hook(U)
        return U, t + dt


class ROS_scheme:
    """docstring for FE_scheme"""

    def __init__(self, model):
        self.gamma = 1 - 1 / 2 * np.sqrt(2)
        self.model = model

    def __call__(self, U, t, dt, pars, hook=lambda x: x):
        J = self.model.J(U, pars)
        J = sps.eye(U.size, format='csc') - self.gamma * dt * J
        luf = sps.linalg.splu(J)
        F = self.model.F(U, pars)
        k1 = luf.solve(dt * F)
        F = self.model.F(U + k1, pars)
        k2 = luf.solve(dt * F - 2 * k1)

        U = U + 3 / 2 * k1 + 1 / 2 * k2
        U = hook(U)
        return U, t + dt


class ROS_vart_scheme:
    """docstring for FE_scheme"""

    def __init__(self, model):
        self.model = model
        self.gamma = 1 - 1 / 2 * np.sqrt(2)
        self.internal_dt = 1E-4

    def one_step(self, U, dt, pars):
        """

        Parameters
        ----------
        U :

        dt :


        Returns
        -------

        """

        err = None
        while (err is None or err > pars['tol']):
            J = self.model.J(U, pars)
            J = sps.eye(U.size, format='csc') - self.gamma * dt * J
            luf = sps.linalg.splu(J)
            F = self.model.F(U, pars)
            k1 = luf.solve(dt * F)
            F = self.model.F(U + k1, pars)
            k2 = luf.solve(dt * F - 2 * k1)

            Ubar = U + k1
            U_new = U + 3 / 2 * k1 + 1 / 2 * k2

            err = norm(U_new - Ubar, ord=np.inf)
            dt = 0.9 * dt * np.sqrt(pars['tol'] / err)
        return U_new, dt, err

    def __call__(self, U, t, dt, pars, hook=lambda x: x):
        self.next_time_step = t + dt
        self.internal_iter = 0
        while True:
            logging.debug(f'ROS_vart, iter {self.internal_iter}, t {t}')
            Unew, dt_calc, self.err = self.one_step(U,
                                                    self.internal_dt,
                                                    pars)
            logging.debug(f'dt computed after err below tol: {dt_calc}')
            t = t + self.internal_dt
            logging.debug(f'ROS_vart, t {t}')
            if dt_calc > dt:
                dt_calc = dt
                logging.debug(f'dt computed bigger than asked, '
                              'falling to dt {dt_calc}')
            dt_new = dt_calc
            if t + dt_calc >= self.next_time_step:
                logging.debug('new t more than next expected time step')
                dt_new = self.next_time_step - t
                logging.debug(f'dt falling to {dt_new}')
            self.internal_dt = dt_new
            U = hook(Unew)
            self.internal_iter += 1
            if np.isclose(t, self.next_time_step):
                self.next_time_step += dt
                self.internal_dt = dt_calc
                return U, self.next_time_step
            if self.internal_iter > pars.get('max_iter',
                                             self.internal_iter + 1):
                raise RuntimeError("Rosebrock internal iteration "
                                   "above max iterations authorized")
            if self.internal_dt < pars.get('dt_min', self.internal_dt * .5):
                raise RuntimeError("Rosebrock internal time step "
                                   "less than authorized")
