#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
import scipy.sparse as sps

from triflow.core.simulation import Simulation
from triflow.core.make_routines import load_routines_fortran
from triflow.misc.misc import coroutine


def rebuild_solver(routine_name):
    solver = Solver(routine_name)
    return solver


class Solver(object):
    """ """

    def __init__(self, routine):
        self.routine_name = None
        if isinstance(routine, str):
            self.routine_name = routine
            routine = load_routines_fortran(routine)
        (self.func_i, self.jacob_i,
         self.U, self.parameters,
         self.helpers,
         ((self.Fbdc, self.Jbdc, self.Hsbdc,
           self.fields_order,
           self.bdc_fsymbols,
           self.bdc_parameters))) = routine
        self.window_range, self.nvar = self.U.shape
        self.fields = np.vectorize(lambda x: str(x).split('_')[0],
                                   otypes=[str])(self.U[0])
        logging.info('Champs: %s' % ' - '.join(self.fields))

    def check_pars(self, pars, parlist):
        """

        Parameters
        ----------
        pars :

        parlist :


        Returns
        -------


        """

        try:
            pars = [pars.get(key.name, 0)
                    for key
                    in parlist]
        except AttributeError:
            pars = [pars.get(key, 0)
                    for key
                    in parlist]
        return pars

    def get_fields(self, flat_data):
        """

        Parameters
        ----------
        flat_data :


        Returns
        -------


        """
        new_data = flat_data.T
        new_data.dtype = [(name, float) for name in self.fields]
        return new_data.T

    def flatten_fields(self, *fields):
        """

        Parameters
        ----------
        *fields :


        Returns
        -------


        """

        flat_data = np.array(fields).flatten('F')
        return flat_data

    def compute_F(self, data, **pars):
        """

        Parameters
        ----------
        data :

        **pars :


        Returns
        -------


        """

        nvar = self.nvar
        window_range = self.window_range
        bdc_range = int((window_range - 1) / 2)
        Nx = int(data.size / nvar)
        Fpars = self.check_pars(pars, self.parameters)
        F = np.zeros(data.shape)
        Ui = np.zeros([nvar * window_range])
        for i in np.arange(bdc_range, Nx - bdc_range):
            Ui[:] = data[(i - bdc_range) * nvar:
                         (i + bdc_range + 1) * nvar]
            Fi = self.func_i(*Ui, *Fpars)
            for ivar in range(nvar):
                F[i * nvar + ivar] = Fi[ivar, 0]
        bdcpars = self.check_pars(pars, self.bdc_parameters)
        bdc_args = (data[:(bdc_range + 2) * nvar].tolist() +
                    data[-(bdc_range + 2) * nvar:].tolist() +
                    bdcpars)

        Fbdc = np.array([fbdc(*bdc_args).squeeze()
                         for fbdc
                         in self.Fbdc]).flatten()

        F[:(bdc_range) * nvar] = Fbdc[:(bdc_range) * nvar]
        F[-(bdc_range) * nvar:] = Fbdc[(bdc_range) * nvar:]
        return F

    def compute_Hs(self, data, **pars):
        """

        Parameters
        ----------
        data :

        **pars :


        Returns
        -------


        """

        nvar = self.nvar
        window_range = self.window_range
        bdc_range = int((window_range - 1) / 2)
        Nx = int(data.size / nvar)
        Hpars = self.check_pars(pars, self.parameters)
        Hs = []
        Ui = np.zeros([nvar * window_range])
        for j, help_i in enumerate(self.helpers):
            H = np.zeros(Nx)
            Ui = np.zeros([nvar * window_range])
            for i in np.arange(bdc_range, Nx - bdc_range):
                Ui[:] = data[(i - bdc_range) * nvar:
                             (i + bdc_range + 1) * nvar]
                Hi = help_i(*Ui, *Hpars)
                H[i] = Hi
            bdcpars = self.check_pars(pars, self.bdc_parameters)
            bdc_args = (data[:(bdc_range + 2) * nvar].tolist() +
                        data[-(bdc_range + 2) * nvar:].tolist() +
                        bdcpars)

            Hbdc = np.array([hbdc(*bdc_args)
                             for hbdc
                             in self.Hsbdc[::-1][j]]).flatten()

            H[:(bdc_range)] = Hbdc[:(bdc_range)]
            H[-(bdc_range):] = Hbdc[(bdc_range):]
            Hs.append(H)
        return Hs

    def compute_J(self, data, **pars):
        """

        Parameters
        ----------
        data :

        **pars :


        Returns
        -------


        """

        nvar = self.nvar
        window_range = self.window_range
        bdc_range = int((window_range - 1) / 2)
        Nx = int(data.size / nvar)
        Jpars = self.check_pars(pars, self.parameters)
        J = np.zeros([nvar * Nx, nvar * Nx])
        Ui = np.zeros([nvar * window_range])
        for i in range(bdc_range, Nx - bdc_range):
            Ui[:] = data[(i - bdc_range) * nvar: (i + bdc_range + 1) * nvar]
            Ji = self.jacob_i(*Ui, *Jpars)
            for ivar in range(nvar):
                J[i * nvar + ivar,
                  nvar * (i - bdc_range):
                  nvar * (i + bdc_range + 1)] = Ji[ivar]
        bdcpars = self.check_pars(pars, self.bdc_parameters)
        bdc_args = (data[:(bdc_range + 2) * nvar].tolist() +
                    data[-(bdc_range + 2) * nvar:].tolist() +
                    bdcpars)

        Jbdc = np.array([jbdc(*bdc_args).squeeze()
                         for jbdc
                         in self.Jbdc]).reshape((bdc_range * 2 * nvar, -1))

        Jbdctopleft = Jbdc[:bdc_range * nvar, :(bdc_range + 2) * nvar]
        Jbdctopright = Jbdc[:bdc_range * nvar, (bdc_range + 2) * nvar:]

        Jbdcbottomleft = Jbdc[bdc_range * nvar:, :(bdc_range + 2) * nvar]
        Jbdcbottomright = Jbdc[bdc_range * nvar:, (bdc_range + 2) * nvar:]

        J[:bdc_range * nvar, :nvar * (bdc_range + 2)] = Jbdctopleft
        J[:bdc_range * nvar, -nvar * (bdc_range + 2):] = Jbdctopright

        J[-bdc_range * nvar:, :nvar * (bdc_range + 2)] = Jbdcbottomleft
        J[-bdc_range * nvar:, -nvar * (bdc_range + 2):] = Jbdcbottomright

        return J

    @coroutine
    def compute_J_sparse(self, data, **pars):
        """

        Parameters
        ----------
        data :

        **pars :


        Returns
        -------


        """

        nvar = self.nvar
        # window_range = self.window_range
        Nx = int(data.size / nvar)

        def init_sparse():
            """ """
            rand_data = np.random.rand(*data.shape)
            random_J = self.compute_J(rand_data,
                                      **pars)
            full_coordinates = np.indices(random_J.shape)

            Jcoordinate_array = [full_coordinate
                                 for full_coordinate
                                 in full_coordinates]

            Jpadded = tuple(Jcoordinate_array)

            maskJ = random_J[Jpadded] != 0
            row_padded = np.indices((nvar * Nx, nvar * Nx))[0][maskJ]
            col_padded = np.indices((nvar * Nx, nvar * Nx))[1][maskJ]
            row = full_coordinates[0][Jpadded][maskJ]
            col = full_coordinates[1][Jpadded][maskJ]
            return row_padded, col_padded, row, col

        row_padded, col_padded, row, col = init_sparse()
        data, t = yield
        while True:
            J = self.compute_J(data,
                               **pars)
            data = J[tuple([row, col])]
            J_sparse = sps.csc_matrix(
                (data, (row_padded, col_padded)),
                shape=(nvar * Nx, nvar * Nx))
            data, t = yield J_sparse

    def start_simulation(self, U0, t0, **pars):
        """

        Parameters
        ----------
        U0 :

        t0 :

        **pars :


        Returns
        -------


        """

        return Simulation(self, U0, t0, **pars)

    def __reduce__(self):
        if self.routine_name is None:
            raise ValueError('cannot pickle not named solver')
        return (rebuild_solver, (self.routine_name,))
