#!/usr/bin/env python
# coding=utf8

import itertools as it
import logging
from threading import Event
from uuid import uuid1

import numpy as np

from triflow.plugins import displays, schemes


def rebuild_simul(id, solver, U, t, display,
                  writers, signals,
                  drivers, internal_iter, err, scheme, i, pars, conf):
    newid = id.split('_')
    if len(newid) == 1:
        newid = '_'.join(newid + ['0'])
    else:
        newid = '_'.join(newid[:-1] + [str(int(newid[-1]) + 1)])
    new_simul = Simulation(solver, U, t,
                           id=newid,
                           **pars)
    new_simul.display = display
    new_simul.writers = writers.copy()
    new_simul.signals = signals.copy()
    new_simul.drivers = drivers.copy()
    new_simul.internal_iter = internal_iter
    new_simul.err = err
    new_simul.scheme = scheme
    new_simul.i = i
    new_simul.conf = conf
    return new_simul


class Simulation(object):
    """ """

    def __init__(self, solver, U0, t0=0, id=None, **pars):
        self.id = str(uuid1()) if id is None else id
        self.solver = solver
        self.pars = pars
        self.nvar = self.solver.nvar
        U0 = solver.flatten_fields(U0)

        logging.debug('U field after flattend: %s' % (' - '.join(map(str,
                                                                     U0.shape)
                                                                 )
                                                      )
                      )

        self.U = U0
        self.t = t0
        self.i = 0
        self.x, self.pars['dx'] = np.linspace(0, pars['L'], pars['Nx'],
                                              retstep=True)
        self.iterator = self.compute()
        self.internal_iter = None
        self.err = None
        self.Jcomp = self.solver.compute_J_sparse(self.U,
                                                  **self.pars)
        self.drivers = []
        self.writers = []
        self.display = displays.simple_display
        self.scheme = schemes.ROS_vart_scheme
        self.stop = Event()
        self.signals = {}
        self.conf = {}

    def compute(self):
        """ """

        nvar = self.nvar
        self.pars['Nx'] = int(self.U.size / nvar)
        self.U = self.hook(self.U)

        display = self._init_display_()
        writers = self._init_writers_()
        numerical_scheme = self._init_scheme_()
        yield next(display)

        for self.i, self.U in enumerate(filter(
                self.filter,
                it.takewhile(self.takewhile,
                             numerical_scheme))):
            self.t += self.pars['dt']
            self.driver(self.t)
            self.write(writers)
            yield next(display)
        self.stop.set()

    def compute_until_finished(self):
        logging.info('simulation %s computing until the end' % self.id)
        self.pars['tmax']
        for iteration in self.iterator:
            logging.info('simulation reached time %.2f, iteration %i' %
                         (self.t, self.i))

    def add_signal(self, field, signal):
        self.signals[field] = signal

    def add_writer(self, writer, replace=True):
        try:
            assert not(writer.writer_type in [writer.writer_type
                                              for writer
                                              in self.writers]), \
                'Already %s writer attached' % writer.writer_type
        except AttributeError:
            logging.warning('writer_type not found')
        except AssertionError:
            logging.warning('Already %s writer attached, replacing..' %
                            writer.writer_type)
            [self.writers.remove(oldwriter)
             for oldwriter in self.writers
             if getattr(oldwriter, 'writer_type', None) == writer.type]
        self.writers.append(writer)

    def _init_writers_(self):
        return [writer(self) for writer in self.writers]

    def _init_display_(self):
        return self.display(self)

    def set_display(self, display):
        self.display = display

    def set_scheme(self, scheme):
        if scheme == 'theta':
            if self.pars['theta'] != 0:
                self.scheme = schemes.BE_scheme
                return
            self.scheme = schemes.FE_scheme
            return
        if scheme == 'BDF':
            self.scheme = schemes.BDF2_scheme
            return
        if scheme == 'BDF-alpha':
            self.scheme = schemes.BDFalpha_scheme
            return
        if scheme == 'SDIRK':
            self.scheme = schemes.SDIRK_scheme
            return
        if scheme == 'ROS':
            self.scheme = schemes.ROS_scheme
            return
        if scheme == 'ROS_vart':
            self.scheme = schemes.ROS_vart_scheme
            return
        if callable(scheme):
            self.scheme = scheme
            return
        raise NotImplementedError('method not implemented')

    def _init_scheme_(self):
        return self.scheme(self)

    def write(self, writers):
        for writer in writers:
            next(writer)

    def driver(self, t):
        """
        Modify the parameters at each internal time steps. The driver have
        to be appened to the attribute drivers.
        Parameters
        ----------
        t : actual time


        Returns
        -------
        None

        """

        for driver in self.drivers:
            driver(self, t)

        for field, signal in self.signals.items():
            self.pars['%sini' % field] = signal(t)

    def takewhile(self, U):
        """

        Parameters
        ----------
        U :


        Returns
        -------
        Stopping condition for the simulation: without overide
        this will raise an error if the film thickness go less than 0 and
        exit when tmax is reached if in the parameters.
        """

        if any(U[::self.nvar] < 0):
            logging.error('h above 0, solver stopping')
            raise RuntimeError('h above 0')
        if self.pars.get('tmax', None) is None:
            return True
        if self.t >= self.pars.get('tmax', None):
            return False
        return True

    def filter(self, U):
        """

        Parameters
        ----------
        U :


        Returns
        -------
        Tell when return the solution to the user.
        Default return all solutions reaching dt.
        """

        return True

    def hook(self, U):
        """

        Parameters
        ----------
        U :


        Returns
        -------

        """

        return U

    def __iter__(self):
        return self.iterator

    def __next__(self):
        return next(self.iterator)

    def copy(self):
        newid = self.id.split('_')
        if len(newid) == 1:
            newid = '_'.join(newid + ['0'])
        else:
            newid = '_'.join(newid[:-1] + [str(int(newid[-1]) + 1)])
        new_simul = Simulation(self.solver, self.U, self.t,
                               id=newid,
                               **self.pars)
        new_simul.display = self.display
        new_simul.writers = self.writers.copy()
        new_simul.signals = self.signals.copy()
        new_simul.drivers = self.drivers.copy()
        new_simul.internal_iter = self.internal_iter
        new_simul.err = self.err
        new_simul.scheme = self.scheme
        new_simul.i = self.i
        new_simul.conf = self.conf
        return new_simul

    def __copy__(self):
        return self.copy()

    def __reduce__(self):

        return rebuild_simul, (self.id, self.solver, self.U, self.t,
                               self.display, self.writers, self.signals,
                               self.drivers, self.internal_iter, self.err,
                               self.scheme, self.i, self.pars, self.conf)
