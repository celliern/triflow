#!/usr/bin/env python3
# coding=utf8

import importlib.util as impu
import itertools as it
import os
import subprocess
from functools import partial
from logging import info
from multiprocessing import Pool

import sympy as sp
from path import Path
from sympy.utilities.autowrap import autowrap
from sympy.utilities.codegen import codegen

import numpy as np
from triflow.misc.misc import (cd, extract_parameters, order_field,
                               write_codegen)
from triflow.models.boundaries import analyse_boundary
from triflow.path_project import fmodel_dir


def make_routines_fortran(model, boundary):
    """

    Parameters
    ----------
    model : a physical model as in triflow.models representing the model
    \partial t U = F(U) : it should be a callable and return the local
    unknowns (needed for the i-st row of the matrix), the function F evaluated
    at the i-st row, the jacobian J evaluated at the i-st row, all the
    physical parameters and helper functions (which can be empty).

    boundary : a boundary as in triflow.models.boundaries, which return
    (Fboundaries, Jboundaries, Hboundaries), the function F, the Jacobian and
    the helpers evaluated at the boundaries of the domain.


    Returns
    -------
    func_i: a Fortran routine which compute F at the row i.
    jacob_i: a Fortran routine which compute J at the row i.
    U: the local unknowns
    parameters: the physical parameters
    helps_i: a Fortran routine which compute the helpers at the row i.
    (Fbounds, Jbounds,
     helps_i,
     fields_order,
     bdc_fsymbols,
     bdc_parameters):
        the routines and informations needed to compute the boundaries.

    """
    U, F, J, pars, Helps = model()
    fields_order = order_field(U)

    window_range, nvar = U.shape

    Fbdcs, Jbdcs, Hsbdcs = boundary((U, F, J, pars, Helps))

    bdc_fsymbols, bdc_parameters = analyse_boundary((Fbdcs, Jbdcs),
                                                    fields_order,
                                                    window_range, nvar)

    info('extraction des paramètres')
    Fparams = extract_parameters(F, U)
    Jparams = extract_parameters(J, U)

    # On fait l'union des deux séries de paramètres (au cas où), et on s'assure
    # de garder la liste dans un ordre cohérent.
    parameters = sorted(list(Fparams.union(Jparams)),
                        key=lambda symbol: symbol.name)
    info("paramètres (ordonnés): " +
         ' - '.join([par.name for par in parameters]))
    info("écriture et compilation des routines Fortran")
    info("membre de gauche")

    func_i = autowrap(F,
                      args=U.flatten().tolist() + parameters)
    info("done")
    info("jacobien...")
    jacob_i = autowrap(J,
                       args=U.flatten().tolist() + parameters)
    info("done")

    info("fonctions d'aide...")
    helps_i = []
    for H in Helps:
        helps_i.append(autowrap(H,
                                args=U.flatten().tolist() + parameters))
    info("done")

    info("conditions limites...")

    Fbounds = []
    for Fbdc in Fbdcs:
        Fbounds.append(autowrap(Fbdc, args=(map(sp.Symbol,
                                                bdc_fsymbols
                                                .flatten().tolist() +
                                                bdc_parameters))))

    Jbounds = []
    for Jbdc in Jbdcs:
        Jbounds.append(autowrap(Jbdc, args=(map(sp.Symbol,
                                                bdc_fsymbols
                                                .flatten().tolist() +
                                                bdc_parameters))))

    info("done")
    return func_i, jacob_i, U, parameters, helps_i, (Fbounds, Jbounds,
                                                     helps_i,
                                                     fields_order,
                                                     bdc_fsymbols,
                                                     bdc_parameters)


def load_routines_fortran(folder: str):
    """

    Parameters
    ----------
    folder : str : the name of the folder where is cached the fortran routine

    Returns
    -------
    func_i: a Fortran routine which compute F at the row i.
    jacob_i: a Fortran routine which compute J at the row i.
    U: the local unknowns
    parameters: the physical parameters
    helps_i: a Fortran routine which compute the helpers at the row i.
    (Fbounds, Jbounds,
     helps_i,
     fields_order,
     bdc_fsymbols,
     bdc_parameters):
        the routines and informations needed to compute the boundaries.

    """

    working_dir = fmodel_dir / folder
    info(working_dir)
    U = np.load(working_dir / 'U_symb.npy')
    parameters = np.load(working_dir / 'pars_symb.npy')
    fields_order = np.load(working_dir / 'field_order.npy')
    bdc_fsymbols = np.load(working_dir / 'bdc_fsymbols.npy')
    bdc_parameters = np.load(working_dir / 'bdc_parameters.npy')

    info("paramètres (ordonnés): " +
         ' - '.join([par.name for par in parameters]))
    try:
        spec = impu.spec_from_file_location("F",
                                            working_dir /
                                            'F.cpython-35m-x86_64-linux-gnu.so')
        func_i = impu.module_from_spec(spec).f
    except ImportError:
        spec = impu.spec_from_file_location("F",
                                            working_dir /
                                            'F.so')
        func_i = impu.module_from_spec(spec).f
    try:
        spec = impu.spec_from_file_location(
            "J", working_dir / 'J.cpython-35m-x86_64-linux-gnu.so')
        jacob_i = impu.module_from_spec(spec).j
    except ImportError:
        spec = impu.spec_from_file_location(
            "J", working_dir / 'J.so')
        jacob_i = impu.module_from_spec(spec).j

    help_routines = [file.namebase for
                     file in
                     working_dir.files()
                     if (file.ext == '.f90' and
                         file.namebase[0] == 'H' and
                         not file.namebase[:4] == 'Hbdc')]
    helps_i = []
    for H in help_routines:
        try:
            spec = impu.spec_from_file_location(
                H, working_dir / '%s.cpython-35m-x86_64-linux-gnu.so' % H)
            help_i = impu.module_from_spec(spec).h
        except ImportError:
            spec = impu.spec_from_file_location(H, working_dir / '%s.so' % H)
            help_i = impu.module_from_spec(spec).h
        helps_i.append(help_i)

    Fbdc_routines = sorted([file.namebase for
                            file in
                            working_dir.files()
                            if (file.ext == '.f90' and
                                file.namebase[:4] == 'Fbdc')])
    Fbounds = []
    for Fbdc in Fbdc_routines:
        try:
            spec = impu.spec_from_file_location(
                Fbdc, working_dir /
                '%s.cpython-35m-x86_64-linux-gnu.so' % Fbdc)
            Fbound = impu.module_from_spec(spec).fbdc
        except ImportError:
            spec = impu.spec_from_file_location(
                Fbdc, working_dir / '%s.so' % Fbdc)
            Fbound = impu.module_from_spec(spec).fbdc
        Fbounds.append(Fbound)

    Jbdc_routines = sorted([file.namebase for
                            file in
                            working_dir.files()
                            if (file.ext == '.f90' and
                                file.namebase[:4] == 'Jbdc')])
    Jbounds = []
    for Jbdc in Jbdc_routines:
        try:
            spec = impu.spec_from_file_location(
                Jbdc, working_dir /
                '%s.cpython-35m-x86_64-linux-gnu.so' % Jbdc)
            Jbound = impu.module_from_spec(spec).jbdc
        except ImportError:
            spec = impu.spec_from_file_location(Jbdc, working_dir /
                                                '%s.so' % Jbdc)
            Jbound = impu.module_from_spec(spec).jbdc
        Jbounds.append(Jbound)

    Hsbounds = []
    Hsbdc_routines = sorted([file.namebase for
                             file in
                             working_dir.files()
                             if (file.ext == '.f90' and
                                 file.namebase[:4] == 'Hbdc')])

    for i in sorted(list(set([Hsbdc_routine[4]
                              for Hsbdc_routine
                              in Hsbdc_routines]))):
        Hbounds = []
        Hbdc_routines = sorted([file.namebase for
                                file in
                                working_dir.files()
                                if (file.ext == '.f90' and
                                    file.namebase[:5] == 'Hbdc%i' % int(i))])
        for Hbdc in Hbdc_routines:
            try:
                spec = impu.spec_from_file_location(
                    Hbdc, working_dir /
                    '%s.cpython-35m-x86_64-linux-gnu.so' % Hbdc)
                Hbound = impu.module_from_spec(spec).hbdc
            except ImportError:
                spec = impu.spec_from_file_location(
                    Hbdc, working_dir /
                    '%s.so' % Hbdc)
                Hbound = impu.module_from_spec(spec).hbdc
            Hbounds.append(Hbound)
        Hsbounds.append(Hbounds)

    return func_i, jacob_i, U, parameters, helps_i, (Fbounds, Jbounds,
                                                     Hsbounds,
                                                     fields_order,
                                                     bdc_fsymbols,
                                                     bdc_parameters)


def comp_function(routine, working_dir: (str, Path) = Path('.')):
    """

    Parameters
    ----------
    routine : the name of the wanted fortran source.

    working_dir: str or Path: the folder where the fortran source is placed.


    Returns
    -------
    None

    """

    fnull = open(os.devnull, 'w')
    with cd(working_dir):
        subprocess.call(["f2py", "-c", "-m",
                         "%s" % routine, "%s.f90" % routine],
                        stdout=fnull)


def compile_routines_fortran(folder: str):
    """Permet de compiler les fonctions fortran contenu dans le dossier folder,
    doit être appelé après la fonction cache_routine_fortran.
    La compilation est faite en //, afin de gagner un peu de temps sur
    les gros modèles.

    Parameters
    ----------
    folder :


    Returns
    -------


    """
    working_dir = fmodel_dir / folder

    partial_comp = partial(comp_function, working_dir=working_dir)
    routines = ([file.namebase for
                 file in
                 working_dir.files()
                 if file.ext == '.f90'])
    p = Pool(len(routines))
    p.map(partial_comp, routines)


def cache_routines_fortran(model, boundary, folder: str):
    """

    Parameters
    ----------
    model :

    boundary :

    folder : str :


    Returns
    -------


    """
    U, F, J, pars, Helps = model()
    fields_order = order_field(U)

    window_range, nvar = U.shape

    Fbdcs, Jbdcs, Hsbdcs = boundary((U, F, J, pars, Helps))

    bdc_fsymbols, bdc_parameters = analyse_boundary((Fbdcs, Jbdcs),
                                                    fields_order,
                                                    window_range, nvar)

    info('extraction des paramètres')
    Fparams = extract_parameters(F, U)
    Jparams = extract_parameters(J, U)

    # On fait l'union des deux séries de paramètres (au cas où), et on s'assure
    # de garder la liste dans un ordre cohérent.
    parameters = sorted(list(Fparams.union(Jparams)),
                        key=lambda symbol: symbol.name)
    info("paramètres (ordonnés): " +
         ' - '.join([par.name for par in parameters]))
    info("écriture et compilation des routines Fortran")
    info("membre de gauche")
    F_matsymb = sp.MatrixSymbol('Fout', *F.shape)
    func_i = codegen(('F', sp.Eq(F_matsymb,
                                 F)),
                     'F95',
                     argument_sequence=(U.flatten().tolist() +
                                        parameters +
                                        [F_matsymb]))

    J_matsymb = sp.MatrixSymbol('Jout', *J.shape)
    jacob_i = codegen(('J', sp.Eq(J_matsymb,
                                  J)),
                      'F95',
                      argument_sequence=(U.flatten().tolist() +
                                         parameters +
                                         [J_matsymb]))

    helps_i = []
    for i, H in enumerate(Helps):
        H_symb = sp.Symbol('Hout')
        help_i = codegen(('H', sp.Eq(H_symb,
                                     H)),
                         'F95',
                         argument_sequence=(U.flatten().tolist() +
                                            parameters +
                                            [H_symb]))
        helps_i.append(help_i)

    Fbounds = []
    for i, Fbdc in enumerate(Fbdcs):
        Fbdc_matsymb = sp.MatrixSymbol('Fbdcout', *Fbdc.shape)
        Fbound = codegen(('Fbdc', sp.Eq(Fbdc_matsymb,
                                        Fbdc)),
                         'F95',
                         argument_sequence=(list(map(sp.Symbol,
                                                     bdc_fsymbols.flatten()
                                                     .tolist() +
                                                     bdc_parameters)) +
                                            [Fbdc_matsymb])
                         )
        Fbounds.append(Fbound)

    Jbounds = []
    for i, Jbdc in enumerate(Jbdcs):
        Jbdc_matsymb = sp.MatrixSymbol('Jbdcout', *Jbdc.shape)
        Jbound = codegen(('Jbdc', sp.Eq(Jbdc_matsymb,
                                        Jbdc)),
                         'F95',
                         argument_sequence=(list(map(sp.Symbol,
                                                     bdc_fsymbols.flatten()
                                                     .tolist() +
                                                     bdc_parameters)) +
                                            [Jbdc_matsymb])
                         )
        Jbounds.append(Jbound)

    Hsbounds = []
    for i, Hbdcs in enumerate(Hsbdcs):
        Hbounds = []
        for j, Hbdc in enumerate(Hbdcs):
            Hbdc_symb = sp.Symbol('Hbdcout')
            Hbound = codegen(('Hbdc', sp.Eq(Hbdc_symb,
                                            Hbdc)),
                             'F95',
                             argument_sequence=(list(map(sp.Symbol,
                                                         bdc_fsymbols.flatten()
                                                         .tolist() +
                                                         bdc_parameters)) +
                                                [Hbdc_symb])
                             )
            Hbounds.append(Hbound)
        Hsbounds.append(Hbounds)

    working_dir = fmodel_dir / folder
    working_dir.rmtree_p()
    working_dir.makedirs()

    codewritter = map(lambda x: write_codegen(x, working_dir),
                      [func_i, jacob_i])

    for i, help_i in enumerate(helps_i):
        it.chain(codewritter,
                 write_codegen(help_i,
                               working_dir,
                               template=lambda filename: "%s_%i.%s" %
                               ('.'.join(filename.split('.')[:-1]),
                                i,
                                filename.split('.')[-1])))
    for i, Fbound in enumerate(Fbounds):
        it.chain(codewritter,
                 write_codegen(Fbound,
                               working_dir,
                               template=lambda filename: "%s_%i.%s" %
                               ('.'.join(filename.split('.')[:-1]),
                                i,
                                filename.split('.')[-1])))
    for i, Jbound in enumerate(Jbounds):
        it.chain(codewritter,
                 write_codegen(Jbound,
                               working_dir,
                               template=lambda filename: "%s_%i.%s" %
                               ('.'.join(filename.split('.')[:-1]),
                                i,
                                filename.split('.')[-1])))
    for i, Hbounds in enumerate(Hsbounds):
        for j, Hbound in enumerate(Hbounds):
            it.chain(codewritter,
                     write_codegen(Hbound,
                                   working_dir,
                                   template=lambda filename:
                                   "%s%i_%i.%s" %
                                   ('.'.join(filename.split('.')[:-1]),
                                    i,
                                    j,
                                    filename.split('.')[-1])))
    list(codewritter)

    np.save(working_dir / 'U_symb', U)
    np.save(working_dir / 'pars_symb', parameters)
    np.save(working_dir / 'field_order', fields_order)
    np.save(working_dir / 'bdc_fsymbols', bdc_fsymbols)
    np.save(working_dir / 'bdc_parameters', bdc_parameters)

    compile_routines_fortran(folder)
