#!/usr/bin/env python
# coding=utf8
import numpy as np


def dumping_hook_h(self, U):
    """

    Parameters
    ----------
    U :


    Returns
    -------

    """

    x = self.x
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
    """

    Parameters
    ----------
    U :


    Returns
    -------

    """

    x = self.x
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
