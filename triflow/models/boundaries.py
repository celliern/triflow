#!/usr/bin/env python3
# coding=utf8

import logging
from logging import debug, info
from functools import reduce

import numpy as np
import sympy as sp


def order_field(U):
    """

    Parameters
    ----------
    U :


    Returns
    -------


    """
    order_field = list(map(lambda y:
                           next(map(lambda x:
                                    str(x).split('_')[0],
                                    y)
                                ),
                           U.T)
                       )
    return order_field


def analyse_boundary(boundaries, fields_order, window_range, nvar):
    """

    Parameters
    ----------
    boundaries :
        param fields_order:
    window_range :
        param nvar:
    fields_order :

    nvar :


    Returns
    -------


    """
    boundaries = np.array(boundaries).flatten()
    total_symbols = sorted(
        map(str,
            reduce(lambda a, b: a.union(b.atoms(sp.Symbol)),
                   boundaries,
                   set()
                   )
            )
    )
    # field_symbols = list(filter(lambda x: "_" in x, total_symbols))
    parameter_symbols = list(filter(lambda x: "_" not in x, total_symbols))
    bdc_range = int((window_range - 1) / 2)
    harmonized_symbol = list(reduce(lambda a, b: a + b, [['%s_%i' % (field, i)
                                                          for i
                                                          in range(bdc_range +
                                                                   2)] +
                                                         ['%s_Nm_%i' % (field,
                                                                        i)
                                                          for i
                                                          in range(bdc_range +
                                                                   1,
                                                                   -1,
                                                                   -1)]
                                                         for field
                                                         in fields_order]))
    info('taille des arguments bdc: %i, %i' % (len(harmonized_symbol),
                                               len(parameter_symbols)))
    # debug(set(harmonized_symbol) &
    #         set(field_symbols))

    bdc_fsymbols, bdc_parameters = (list(harmonized_symbol),
                                    list(parameter_symbols))

    bdc_fsymbols = np.array(bdc_fsymbols).reshape((nvar * 2, -1))
    idx_left = list(filter(lambda x: x % 2 == 0, range(len(bdc_fsymbols))))
    idx_right = list(filter(lambda x: x % 2 != 0, range(len(bdc_fsymbols))))
    bdc_fsymbols = np.concatenate((bdc_fsymbols[idx_left].T.flatten(),
                                   bdc_fsymbols[idx_right].T.flatten()))
    bdc_parameters = sorted(bdc_parameters)

    return bdc_fsymbols, bdc_parameters


def periodic_boundary(model, **pars):
    """

    Parameters
    ----------
    model :
        param **pars:
    **pars :


    Returns
    -------


    """
    U, F, J, pars, helps = model
    fields_order = order_field(U)
    window_range, nvar = U.shape

    def subs_BG(pos):
        """

        Parameters
        ----------
        pos :


        Returns
        -------


        """
        subs_dict = {}
        for i, var in enumerate(U):
            for field in var:
                pos_i = pos + i - 2
                if pos_i < 0:
                    subs_dict[field] = sp.Symbol('%s_Nm_%i' %
                                                 (str(field).split('_')[0],
                                                  np.abs(pos_i + 1)))
                else:
                    subs_dict[field] = sp.Symbol('%s_%i' %
                                                 (str(field).split('_')[0],
                                                  np.abs(pos_i)))
        return subs_dict

    def subs_FG(pos):
        """

        Parameters
        ----------
        pos :


        Returns
        -------


        """
        subs_dict = {}
        for i, var in enumerate(U):
            for field in var:
                pos_i = pos + i - 2
                if pos_i > 0:
                    subs_dict[field] = sp.Symbol('%s_%i' %
                                                 (str(field).split('_')[0],
                                                  np.abs(pos_i - 1)))
                else:
                    subs_dict[field] = sp.Symbol('%s_Nm_%i' %
                                                 (str(field).split('_')[0],
                                                  np.abs(pos_i)))
        return subs_dict

    Fboundaries = [F.subs(subs_BG(0)), F.subs(subs_BG(1)),
                   F.subs(subs_FG(-1)), F.subs(subs_FG(0))]

    Hboundaries = [[H.subs(subs_BG(0)), H.subs(subs_BG(1)),
                    H.subs(subs_FG(-1)), H.subs(subs_FG(0))] for H in helps]

    fsymbols, psymbols = analyse_boundary((Fboundaries, ),
                                          fields_order, window_range, nvar)

    Jboundaries = []
    for Fboundary in Fboundaries:
        Jboundaries.append(sp.Matrix(np.array([Fboundary.diff(sp.Symbol(u))
                                               .tolist()
                                               for u
                                               in fsymbols]).squeeze()
                                     .tolist()).T)

    return (Fboundaries, Jboundaries, Hboundaries)


def openflow_boundary(model):
    """

    Parameters
    ----------
    model :


    Returns
    -------


    """
    U, F, J, pars, helps = model
    fields_order = order_field(U)
    window_range, nvar = U.shape
    dx = sp.Symbol('dx')

    def subs_BG(pos):
        """

        Parameters
        ----------
        pos :


        Returns
        -------


        """
        subs_dict = {}
        for i, var in enumerate(U):
            for field in var:
                pos_i = pos + i - 2
                if pos_i < 0:
                    subs_dict[field] = (sp.Symbol('%sini' %
                                                  str(field).split('_')[0]))
                else:
                    subs_dict[field] = sp.Symbol('%s_%i' %
                                                 (str(field).split('_')[0],
                                                  np.abs(pos_i)))
        return subs_dict

    def subs_FG(pos):
        """

        Parameters
        ----------
        pos :


        Returns
        -------


        """
        subs_dict = {}
        for i, var in enumerate(U):
            for field in var:
                pos_i = pos + i - 2
                if pos_i > 0:
                    subs_dict[field] = sp.Symbol('%s_Nm_%i' %
                                                 (str(field)
                                                  .split('_')[0],
                                                  np.abs(pos_i)))
                else:
                    subs_dict[field] = sp.Symbol('%s_Nm_%i' %
                                                 (str(field).split('_')[0],
                                                  np.abs(pos_i)))
        return subs_dict

    # Our goal is to 'shift' the variables for the 2 first problematics points

    Fboundaries = [F.subs(subs_BG(0)), F.subs(subs_BG(1)),
                   sp.Matrix([(sp.Symbol('%s_Nm_2' % field) -
                               sp.Symbol('%s_Nm_1' % field)) / dx
                              for field in fields_order]),
                   sp.Matrix([(sp.Symbol('%s_Nm_1' % field) -
                               sp.Symbol('%s_Nm_0' % field)) / dx
                              for field in fields_order])]

    Hboundaries = [[H.subs(subs_BG(0)), H.subs(subs_BG(1)),
                    H.subs(subs_FG(-1)), H.subs(subs_FG(0))] for H in helps]

    fsymbols, psymbols = analyse_boundary((Fboundaries, ),
                                          fields_order, window_range, nvar)
    Jboundaries = []
    for Fboundary in Fboundaries:
        Jboundaries.append(sp.Matrix(np.array([Fboundary.diff(sp.Symbol(u))
                                               .tolist()
                                               for u
                                               in fsymbols]).squeeze()
                                     .tolist()).T)
    return (Fboundaries, Jboundaries, Hboundaries)


if __name__ == '__main__':
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
