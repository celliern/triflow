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


def FE_scheme(simul):
    """ """

    solv = ode(lambda t, x: simul.solver.compute_F(x,
                                                   **simul.pars))
    solv.set_integrator('dopri5')
    solv.set_initial_value(simul.U)
    while solv.successful:
        U = solv.integrate(simul.t + simul.pars['dt'])
        U = simul.hook(U)
        yield U


def BDF2_scheme(simul):
    """ """

    U = simul.U
    U = simul.hook(U)
    Id = sps.identity(simul.nvar * simul.pars['Nx'],
                      format='csc')
    Uhist = deque([], 2)
    Uhist.append(U.copy())
    simul.F = F = simul.solver.compute_F(U,
                                         **simul.pars)
    simul.J = J = simul.solver.compute_J_sparse(U,
                                                **simul.pars)
    B = simul.pars['dt'] * (F - J @ U) + U
    J = (Id - simul.pars['dt'] * J)

    U = sps.linalg.lgmres(J, B, x0=U)[0]
    U = simul.hook(U)
    Uhist.append(U.copy())
    yield U
    while True:
        Un = Uhist[-1]
        Unm1 = Uhist[-2]
        dt = simul.pars['dt']
        simul.F = F = simul.solver.compute_F(Un,
                                             **simul.pars)
        simul.J = J = simul.solver.compute_J_sparse(U,
                                                    **simul.pars)
        B = ((4 / 3 * Id - 2 / 3 * dt * J) @ Un -
             1 / 3 * Unm1 +
             2 / 3 * dt * F)
        J = (Id -
             2 / 3 * dt * J)
        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = simul.hook(U)
        Uhist.append(U.copy())
        yield U


def BDFalpha_scheme(simul):
    """ """
    Id = sps.identity(simul.nvar * simul.pars['Nx'],
                      format='csc')
    Uhist = deque([], 2)
    Uhist.append(simul.U.copy())

    simul.F = F = simul.solver.compute_F(simul.U,
                                         **simul.pars)
    U = simul.hook(simul.U)
    simul.J = J = simul.solver.compute_J_sparse(U,
                                                **simul.pars)
    B = simul.pars['dt'] * (F - J @ simul.U) + U
    J = (Id - simul.pars['dt'] * J)

    U = sps.linalg.lgmres(J, B, x0=U)[0]
    U = simul.hook(U)
    Uhist.append(U.copy())
    yield U
    while True:
        alpha = simul.pars['alpha']
        Un = Uhist[-1]
        Unm1 = Uhist[-2]
        dt = simul.pars['dt']
        simul.F = F = simul.solver.compute_F(Un,
                                             **simul.pars)
        simul.J = J = simul.solver.compute_J_sparse(U,
                                                    **simul.pars)
        B = ((Id + alpha * Id) @ (2 * Id - dt * J) @ Un -
             (Id / 2 + alpha * Id) @ Unm1 +
             dt * F)
        J = (alpha + 3 / 2) * Id - dt * (1 + alpha) * J
        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = simul.hook(U)
        Uhist.append(U.copy())
        yield U


def BE_scheme(simul):
    """ """

    U = simul.U
    U = simul.hook(U)
    while True:
        dt = simul.pars['dt']
        simul.F = F = simul.solver.compute_F(U,
                                             **simul.pars)
        simul.J = J = simul.solver.compute_J_sparse(U,
                                                    **simul.pars)
        B = dt * (F -
                  simul.pars['theta'] * J @ U) + U
        J = (sps.identity(simul.nvar * simul.pars['Nx'],
                          format='csc') -
             simul.pars['theta'] * dt * J)
        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = simul.hook(U)
        yield U


def ROS_scheme(simul):
    """DOI: 10.1007/s10543-006-0095-7
    A multirate time stepping strategy
    for stiff ordinary differential equation
    V. Savcenco and al.

    Parameters
    ----------

    Returns
    -------


    """

    U = simul.U
    U = simul.hook(U)
    gamma = 1 - 1 / 2 * np.sqrt(2)
    while True:
        dt = simul.pars['dt']
        simul.J = J = simul.solver.compute_J_sparse(U,
                                                    **simul.pars)
        J = sps.eye(U.size, format='csc') - gamma * dt * J
        luf = sps.linalg.splu(J)
        F = simul.solver.compute_F(U, **simul.pars)
        k1 = luf.solve(dt * F)
        F = simul.solver.compute_F(U + k1, **simul.pars)
        k2 = luf.solve(dt * F - 2 * k1)

        U = U + 3 / 2 * k1 + 1 / 2 * k2
        U = simul.hook(U)
        yield U


def ROS_vart_scheme(simul):
    """DOI: 10.1007/s10543-006-0095-7
    A multirate time stepping strategy
    for stiff ordinary differential equation
    V. Savcenco and al.

    Parameters
    ----------

    Returns
    -------


    """

    U = simul.U
    U = simul.hook(U)
    gamma = 1 - 1 / 2 * np.sqrt(2)
    t = simul.t

    dt = 1E-4
    next_time_step = t + simul.pars['dt']

    # Uhist = deque([U], 3)
    # thist = deque([t], 3)

    def one_step(U, dt):
        """

        Parameters
        ----------
        U :

        dt :


        Returns
        -------

        """

        err = None
        while (err is None or err > simul.pars['tol']):
            simul.J = J = simul.solver.compute_J_sparse(U,
                                                        **simul.pars)
            J = sps.eye(U.size, format='csc') - gamma * dt * J
            luf = sps.linalg.splu(J)
            F = simul.solver.compute_F(U, **simul.pars)
            k1 = luf.solve(dt * F)
            F = simul.solver.compute_F(U + k1, **simul.pars)
            k2 = luf.solve(dt * F - 2 * k1)

            Ubar = U + k1
            U_new = U + 3 / 2 * k1 + 1 / 2 * k2

            err = norm(U_new - Ubar, ord=np.inf)
            dt = 0.9 * dt * np.sqrt(simul.pars['tol'] / err)
        return U_new, dt, err
    simul.internal_iter = 0
    while True:
        Unew, dt_calc, simul.err = one_step(U, dt)
        t = t + dt
        if dt_calc > simul.pars['dt']:
            dt_calc = simul.pars['dt']
        dt_new = dt_calc
        if t + dt_calc >= next_time_step:
            dt_new = next_time_step - t
        dt = dt_new
        U = simul.hook(Unew)
        simul.driver(t)
        simul.internal_iter += 1
        if np.isclose(t, next_time_step):
            next_time_step += simul.pars['dt']
            dt = dt_calc
            yield U
            simul.internal_iter = 0
        if simul.internal_iter > simul.pars.get('max_iter',
                                                simul.internal_iter + 1):
            raise RuntimeError("Rosebrock internal iteration "
                               "above max iterations authorized")
        if dt < simul.pars.get('dt_min',
                               dt * .5):
            raise RuntimeError("Rosebrock internal time step "
                               "less than authorized")


def ROS_mvart_scheme(simul):
    """DOI: 10.1007/s10543-006-0095-7
    A multirate time stepping strategy
    for stiff ordinary differential equation
    V. Savcenco and al.

    Parameters
    ----------

    Returns
    -------


    """

    U = simul.U
    U = simul.hook(U)
    gamma = 1 - 1 / 2 * np.sqrt(2)
    t = simul.t

    dt = 1E1
    next_time_step = t + simul.pars['dt']

    def one_step(U, dt):
        simul.J = J = simul.solver.compute_J_sparse(U,
                                                    **simul.pars)
        J = sps.eye(U.size, format='csc') - gamma * dt * J
        luf = sps.linalg.splu(J)
        F = simul.solver.compute_F(U, **simul.pars)
        k1 = luf.solve(dt * F)
        F = simul.solver.compute_F(U + k1, **simul.pars)
        k2 = luf.solve(dt * F - 2 * k1)

        Ubar = U + k1
        U_new = U + 3 / 2 * k1 + 1 / 2 * k2

        err = U_new - Ubar
        return U_new, dt, err

    def localize_bad_nodes(err):
        return np.where(err > simul.pars['tol'])

    simul.internal_iter = 0
    while True:
        Unew, dt, err = one_step(U, dt)
        components_to_refine = localize_bad_nodes(err)
        yield U
