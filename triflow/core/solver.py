#!/usr/bin/env python
# coding=utf8

import logging

import numpy as np
import scipy.sparse as sps

from triflow.core.simulation import Simulation
from triflow.core.make_routines import load_routines_fortran
from triflow.misc.misc import coroutine


logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


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
        for i in np.arange(bdc_range, Nx - bdc_range):
            Fi = self.func_i(*data[(i - bdc_range) * nvar:
                                   (i + bdc_range + 1) * nvar], *Fpars)
            F[i * nvar: (i + 1) * nvar] = Fi[:, 0]
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
        Hs = {}
        for help_name, help_i in self.helpers.items():
            H = np.zeros(Nx)
            for i in np.arange(bdc_range, Nx - bdc_range):
                Hi = help_i(*data[(i - bdc_range) * nvar:
                                  (i + bdc_range + 1) * nvar], *Hpars)
                H[i] = Hi
            bdcpars = self.check_pars(pars, self.bdc_parameters)
            bdc_args = (data[:(bdc_range + 2) * nvar].tolist() +
                        data[-(bdc_range + 2) * nvar:].tolist() +
                        bdcpars)

            Hbdc = np.array([hbdc(*bdc_args)
                             for hbdc
                             in self.Hsbdc[help_name][::-1]]).flatten()

            H[:(bdc_range)] = Hbdc[:(bdc_range)]
            H[-(bdc_range):] = Hbdc[(bdc_range):]
            Hs[help_name] = H
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
        for i in range(bdc_range, Nx - bdc_range):
            Ji = self.jacob_i(*data[(i - bdc_range) * nvar:
                                    (i + bdc_range + 1) * nvar],
                              *Jpars)
            J[i * nvar: (i + 1) * nvar,
              nvar * (i - bdc_range):
              nvar * (i + bdc_range + 1)] = Ji
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
        window_range = self.window_range
        bdc_range = int((window_range - 1) / 2)
        Nx = int(data.size / nvar)
        Jpars = self.check_pars(pars, self.parameters)

        def mapping(i):
            Ji = self.jacob_i(*data[(i - bdc_range) * nvar:
                                    (i + bdc_range + 1) * nvar],
                              *Jpars)
            indexes, columns = np.meshgrid(
                np.arange(i * nvar,
                          (i + 1) * nvar),
                np.arange(nvar * (i - bdc_range),
                          nvar * (i + bdc_range + 1)),
                indexing='ij')
            return Ji.flatten(), indexes.flatten(), columns.flatten()

        J_list, indexes_list, columns_list = map(np.concatenate,
                                                 zip(
                                                     *map(mapping,
                                                          range(bdc_range,
                                                                Nx - bdc_range)
                                                          )
                                                 )
                                                 )

        bdcpars = self.check_pars(pars, self.bdc_parameters)
        bdc_args = (data[:(bdc_range + 2) * nvar].tolist() +
                    data[-(bdc_range + 2) * nvar:].tolist() +
                    bdcpars)

        Jbdc = np.array([jbdc(*bdc_args).squeeze()
                         for jbdc
                         in self.Jbdc]).reshape((bdc_range * 2 * nvar, -1))

        Jbdctopleft = Jbdc[:bdc_range * nvar,
                           :(bdc_range + 2) * nvar].flatten()
        Jbdctopright = Jbdc[:bdc_range * nvar,
                            (bdc_range + 2) * nvar:].flatten()

        Jbdcbottomleft = Jbdc[bdc_range * nvar:,
                              :(bdc_range + 2) * nvar].flatten()
        Jbdcbottomright = Jbdc[bdc_range * nvar:,
                               (bdc_range + 2) * nvar:].flatten()

        J_list = np.append(J_list, Jbdctopleft)
        J_list = np.append(J_list, Jbdctopright)
        J_list = np.append(J_list, Jbdcbottomleft)
        J_list = np.append(J_list, Jbdcbottomright)

        indexes, columns = np.meshgrid(
            np.arange(0, bdc_range * nvar),
            np.arange(0, nvar * (bdc_range + 2)),
            indexing='ij')
        indexes_list = np.append(indexes_list, indexes)
        columns_list = np.append(columns_list, columns)

        indexes, columns = np.meshgrid(np.arange(0, bdc_range * nvar),
                                       np.arange(Nx * nvar - nvar *
                                                 (bdc_range + 2),
                                                 Nx * nvar),
                                       indexing='ij')
        indexes_list = np.append(indexes_list, indexes)
        columns_list = np.append(columns_list, columns)

        indexes, columns = np.meshgrid(np.arange(Nx * nvar - bdc_range * nvar,
                                                 Nx * nvar),
                                       np.arange(0, nvar * (bdc_range + 2)),
                                       indexing='ij')
        indexes_list = np.append(indexes_list, indexes)
        columns_list = np.append(columns_list, columns)

        indexes, columns = np.meshgrid(np.arange(Nx * nvar - bdc_range * nvar,
                                                 Nx * nvar),
                                       np.arange(Nx * nvar - nvar *
                                                 (bdc_range + 2), Nx * nvar),
                                       indexing='ij')
        indexes_list = np.append(indexes_list, indexes)
        columns_list = np.append(columns_list, columns)
        J = sps.coo_matrix((J_list, (indexes_list, columns_list)),
                           shape=(Nx * nvar, Nx * nvar),
                           dtype=float)

        return J.tocsr()

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
