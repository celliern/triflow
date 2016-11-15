#!/usr/bin/env python
# coding=utf8

import itertools as it
import logging
from collections import deque
from logging import debug, error, info

import numpy as np
import scipy.sparse as sps
from scipy.linalg import norm
from scipy.integrate import ode


class Simulation(object):

    def __init__(self, solver, U0, t0, **pars):
        self.solver = solver
        self.pars = pars
        self.nvar = self.solver.nvar
        self.U = U0
        self.t = t0
        self.iterator = self.compute()
        self.internal_iter = None
        self.err = None
        self.drivers = []

    def FE_scheme(self):
        solv = ode(lambda t, x: self.solver.compute_F(x,
                                                      **self.pars))
        solv.set_integrator('dopri5')
        solv.set_initial_value(self.U)
        while solv.successful:
            U = solv.integrate(self.t + self.pars['dt'])
            U = self.hook(U)
            yield U

    def BDF2_scheme(self):
        U = self.U
        U = self.hook(U)
        Id = sps.identity(self.nvar * self.pars['Nx'],
                          format='csc')
        Uhist = deque([], 2)
        Uhist.append(U.copy())
        Jcomp = self.solver.compute_J_sparse(U,
                                             **self.pars)
        next(Jcomp)
        self.F = F = self.solver.compute_F(U,
                                           **self.pars)
        J = Jcomp.send((U, self.t))
        B = self.pars['dt'] * (F - J @ U) + U
        J = (Id - self.pars['dt'] * J)

        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = self.hook(U)
        Uhist.append(U.copy())
        yield U
        while True:
            Un = Uhist[-1]
            Unm1 = Uhist[-2]
            dt = self.pars['dt']
            self.F = F = self.solver.compute_F(Un,
                                               **self.pars)
            J = Jcomp.send((Un, self.t))
            B = ((4 / 3 * Id - 2 / 3 * dt * J) @ Un -
                 1 / 3 * Unm1 +
                 2 / 3 * dt * F)
            J = (Id -
                 2 / 3 * dt * J)
            U = sps.linalg.lgmres(J, B, x0=U)[0]
            U = self.hook(U)
            Uhist.append(U.copy())
            yield U

    def BDFalpha_scheme(self):
        Id = sps.identity(self.nvar * self.pars['Nx'],
                          format='csc')
        Uhist = deque([], 2)
        Uhist.append(self.U.copy())
        Jcomp = self.solver.compute_J_sparse(self.U,
                                             **self.pars)
        next(Jcomp)
        self.F = F = self.solver.compute_F(self.U,
                                           **self.pars)
        U = self.hook(self.U)
        J = Jcomp.send((U, self.t))
        B = self.pars['dt'] * (F - J @ self.U) + U
        J = (Id - self.pars['dt'] * J)

        U = sps.linalg.lgmres(J, B, x0=U)[0]
        U = self.hook(U)
        Uhist.append(U.copy())
        yield U
        while True:
            alpha = self.pars['alpha']
            Un = Uhist[-1]
            Unm1 = Uhist[-2]
            dt = self.pars['dt']
            self.F = F = self.solver.compute_F(Un,
                                               **self.pars)
            J = Jcomp.send((Un, self.t))
            B = ((Id + alpha * Id) @ (2 * Id - dt * J) @ Un -
                 (Id / 2 + alpha * Id) @ Unm1 +
                 dt * F)
            J = (alpha + 3 / 2) * Id - dt * (1 + alpha) * J
            U = sps.linalg.lgmres(J, B, x0=U)[0]
            U = self.hook(U)
            Uhist.append(U.copy())
            yield U

    def BE_scheme(self):
        U = self.U
        U = self.hook(U)
        Jcomp = self.solver.compute_J_sparse(U,
                                             **self.pars)
        next(Jcomp)
        while True:
            dt = self.pars['dt']
            self.F = F = self.solver.compute_F(U,
                                               **self.pars)
            J = Jcomp.send((U, self.t))
            B = dt * (F -
                      self.pars['theta'] * J @ U) + U
            J = (sps.identity(self.nvar * self.pars['Nx'],
                              format='csc') -
                 self.pars['theta'] * dt * J)
            U = sps.linalg.lgmres(J, B, x0=U)[0]
            U = self.hook(U)
            yield U

    def SDIRK_scheme(self):
        """
        Implementation of Singly Diagonally Implicit Runge-Kutta method
        with constant step sizes.
        Josefine Stal
        Lund University
        """

        class RKscheme(object):
            A = np.array([[1 / 4, 0, 0, 0, 0],
                          [1 / 2, 1 / 4, 0, 0, 0],
                          [17 / 50, -1 / 25, 1 / 4, 0, 0],
                          [371 / 1360, -137 / 2720, 15 / 544, 1 / 4, 0],
                          [25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4]])
            b = np.array([25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4])
            c = np.array([1 / 4, 3 / 4, 11 / 20, 1 / 2, 1])

        scheme = RKscheme()
        U = self.U
        U = self.hook(U)
        Jcomp = self.solver.compute_J_sparse(U,
                                             **self.pars)
        next(Jcomp)

        s = scheme.b.size
        m = U.size

        def F(stage_derivative, t, U):
            """
            Returns the subtraction Y’_{i}-f(t_{n}+c_{i}*h, Y_{i}),
            where Y are
            the stage values, Y’ the stage derivatives and f the
            function of
            the IVP y’=f(t,y) that should be solved by the RK-method.
            Parameters:
            -------------
            stageDer = initial guess of the stage derivatives Y’
            t0 = float, current timestep
            y0 = 1 x m vector, the last solution y_n. Where m is the
            length
            of the initial condition y_0 of the IVP.
            """
            stage_derivative_new = np.empty((s, m))
            for i in np.arange(s):  # iterate over all stageDer
                stage_value = U + np.array([self.pars['dt'] *
                                            scheme.A[i, :] @
                                            stage_derivative[:, j]
                                            for j in np.arange(m)
                                            ])
                stage_derivative_new[i, :] = self.solver.compute_F(stage_value,
                                                                   **self.pars)
            return stage_derivative - stage_derivative_new

        def phi_newtonstep(t, U, init_value, J, lufactor):
            """
            Takes one Newton step by solvning
            G’(Y_i)(Y^(n+1)_i-Y^(n)_i)=-G(Y_i)
            where
            G(Y_i) = Y_i - haY’_i - y_n - h*sum(a_{ij}*Y’_j) for
            j=1,...,i-1
            Parameters:
            -------------
            t0 = float, current timestep
            y0 = 1 x m vector, the last solution y_n. Where m is the
            length
            of the initial condition y_0 of the IVP.
            initVal = initial guess for the Newton iteration
            luFactor = (lu, piv) see documentation for linalg.lu_factor
            Returns:
            The difference Y^(n+1)_i-Y^(n)_i
            """
            def to_solve(x):
                Fs = -F(x, t, U)
                x = np.zeros((s, m))
                for i in np.arange(s):  # solving the s mxm systems
                    p01 = Fs[i]
                    p02 = dt * np.sum([scheme.A[i, j] * (J @ x[j])
                                       for j
                                       in np.arange(1, i - 1)],
                                      axis=0).squeeze()
                    rhs = p01 + p02
                    d = lufactor.solve(rhs)
                    x[i] = d
                return x
            x = to_solve(init_value)
            return init_value + np.array(x), norm(x)

        def phi_solve(t, U, init_value, J):
            """
            This function solves F(Y_i)=0 by solving s systems of size m
            x m each.
            Newton’s method is used with an initial guess initVal.
            Parameters:
            -------------
            t0 = float, current timestep
            y0 = 1 x m vector, the last solution y_n. Where m is the
            length
            of the initial condition y_0 of the IVP.
            initVal = initial guess for the Newton iteration
            J = m x m matrix, the Jacobian matrix of f() evaluated in y_i
            M = maximal number of Newton iterations
            Returns:
            -------------
            The stage derivative Y’_i
            """
            M = self.pars['N_iter']
            alpha = scheme.A[0, 0]
            A = sps.eye(m, format='csc') - dt * alpha * J
            luf = sps.linalg.splu(A)
            for i in range(M):
                init_value, norm_d = phi_newtonstep(
                    t, U, init_value, J, luf)
                if norm_d < self.pars['tol_iter']:
                    # info('The Newton iteration converge after %i' % i)
                    break
                elif i == M - 1:
                    raise ValueError('The Newton iteration did not'
                                     ' converge, final error : %e' % norm_d)
            return init_value

        def phi(t, U):
            stage_derivative = np.array(s *
                                        [self.solver.compute_F(U,
                                                               **self.pars)])
            J = Jcomp.send((U, t))
            stage_value = phi_solve(t, U, stage_derivative, J)
            return np.array([
                scheme.b @ stage_value[:, j]
                for j in range(m)
            ])

        while True:
            dt = self.pars['dt']
            U = U + dt * phi(self.t, U)
            yield U, dt

    def ROS_scheme(self):
        """
        DOI: 10.1007/s10543-006-0095-7
        A multirate time stepping strategy
        for stiff ordinary differential equation
        V. Savcenco and al.
        """

        U = self.U
        U = self.hook(U)
        Jcomp = self.solver.compute_J_sparse(U,
                                             **self.pars)
        next(Jcomp)
        gamma = 1 - 1 / 2 * np.sqrt(2)
        while True:
            dt = self.pars['dt']
            J = Jcomp.send((U, self.t))
            J = sps.eye(U.size, format='csc') - gamma * dt * J
            luf = sps.linalg.splu(J)
            F = self.solver.compute_F(U, **self.pars)
            k1 = luf.solve(dt * F)
            F = self.solver.compute_F(U + k1, **self.pars)
            k2 = luf.solve(dt * F - 2 * k1)

            U = U + 3 / 2 * k1 + 1 / 2 * k2
            U = self.hook(U)
            yield U

    def ROS_vart_scheme(self):
        """
        DOI: 10.1007/s10543-006-0095-7
        A multirate time stepping strategy
        for stiff ordinary differential equation
        V. Savcenco and al.
        """

        U = self.U
        U = self.hook(U)
        Jcomp = self.solver.compute_J_sparse(U,
                                             **self.pars)
        next(Jcomp)
        gamma = 1 - 1 / 2 * np.sqrt(2)
        t = self.t

        dt = 1E-4
        next_time_step = t + self.pars['dt']

        # Uhist = deque([U], 3)
        # thist = deque([t], 3)

        def one_step(U, dt):
            err = None
            while (err is None or err > self.pars['tol']):
                J = Jcomp.send((U, t))
                J = sps.eye(U.size, format='csc') - gamma * dt * J
                luf = sps.linalg.splu(J)
                F = self.solver.compute_F(U, **self.pars)
                k1 = luf.solve(dt * F)
                F = self.solver.compute_F(U + k1, **self.pars)
                k2 = luf.solve(dt * F - 2 * k1)

                Ubar = U + k1
                U = U + 3 / 2 * k1 + 1 / 2 * k2

                err = norm(U - Ubar, ord=np.inf)
                dt = 0.9 * dt * np.sqrt(self.pars['tol'] / err)
            return U, dt, err
        self.internal_iter = 0
        while True:
            Unew, dt_calc, self.err = one_step(U, dt)
            t = t + dt
            if dt_calc > self.pars['dt']:
                dt_calc = self.pars['dt']
            dt_new = dt_calc
            if t + dt_calc >= next_time_step:
                dt_new = next_time_step - t
            dt = dt_new
            U = self.hook(Unew)
            self.driver(t)
            self.internal_iter += 1
            if np.isclose(t, next_time_step):
                next_time_step += self.pars['dt']
                dt = dt_calc
                yield U
                self.internal_iter = 0
            if self.internal_iter > self.pars.get('max_iter',
                                                  self.internal_iter + 1):
                raise RuntimeError("Rosebrock internal iteration "
                                   "above max iterations authorized")
            if dt < self.pars.get('dt_min',
                                  dt * .5):
                raise RuntimeError("Rosebrock internal time step "
                                   "less than authorized")

    def compute(self):
        nvar = self.nvar
        self.pars['Nx'] = int(self.U.size / nvar)
        self.U = self.hook(self.U)

        if self.pars['method'] == 'theta':
            if self.pars['theta'] != 0:
                numerical_scheme = self.BE_scheme()
            else:
                numerical_scheme = self.FE_scheme()
        elif self.pars['method'] == 'BDF':
            numerical_scheme = self.BDF2_scheme()
        elif self.pars['method'] == 'BDF-alpha':
            numerical_scheme = self.BDFalpha_scheme()
        elif self.pars['method'] == 'SDIRK':
            numerical_scheme = self.SDIRK_scheme()
        elif self.pars['method'] == 'ROS':
            numerical_scheme = self.ROS_scheme()
        elif self.pars['method'] == 'ROS_vart':
            numerical_scheme = self.ROS_vart_scheme()
        else:
            raise NotImplementedError('method not implemented')
        for self.i, self.U in enumerate(it.takewhile(self.takewhile,
                                                     numerical_scheme)):
            self.t += self.pars['dt']
            self.driver(self.t)
            self.writter(self.t, self.U)
            yield self.display()

    def writter(self, t, U):
        pass

    def driver(self, t):
        """
        Like a hook, called after every successful step:
        this is what is returned to the user after each iteration. Can be
        easily replaced to an other driver, for example in order
        to manage the time step or boundaries.
        """
        for driver in self.drivers:
            driver(self, t)

    def display(self):
        """
        Like a hook, called after every successful step:
        this is what is returned to the user after each iteration. Can be
        easily replaced to an other driver.
        """
        return self.solver.get_fields(self.U), self.t

    def takewhile(self, U):
        if True in (U[::self.nvar] < 0):
            error('h above 0, solver stopping')
            raise RuntimeError('h above 0')
        return True

    def hook(self, U):
        return U

    def dumping_hook_h(self, U):
        x = np.linspace(0, self.pars['Nx'] * self.pars['dx'], self.pars['Nx'])
        U[::self.nvar] = (U[::self.nvar] *
                          (-(np.tanh((x - self.pars['dx'] *
                                      self.pars['Nx']) /
                                     (self.pars['dx'] *
                                      self.pars['Nx'] / 10)) + 1) /
                           2 + 1) +
                          ((np.tanh((x - self.pars['dx'] *
                                     self.pars['Nx']) /
                                    (self.pars['dx'] *
                                     self.pars['Nx'] / 10)) + 1) / 2) *
                          self.pars['hhook'])
        return U

    def dumping_hook_q(self, U):
        x = np.linspace(0, self.pars['Nx'] * self.pars['dx'], self.pars['Nx'])
        U[1::self.nvar] = (U[1::self.nvar] *
                           (-(np.tanh((x - self.pars['dx'] *
                                       self.pars['Nx']) /
                                      (self.pars['dx'] *
                                       self.pars['Nx'] / 10)) + 1) /
                            2 + 1) +
                           ((np.tanh((x - self.pars['dx'] *
                                      self.pars['Nx']) /
                                     (self.pars['dx'] *
                                      self.pars['Nx'] / 10)) + 1) / 2) *
                           self.pars['qhook'])
        return U

    def __iter__(self):
        return self.iterator

    def __next__(self):
        return next(self.iterator)
