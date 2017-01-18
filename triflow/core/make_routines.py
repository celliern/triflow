#!/usr/bin/env python3
# coding=utf8

import itertools as it
import logging
import os
import subprocess
from functools import partial
from multiprocessing import Pool

import importlib.util as impu
import numpy as np
import sympy as sp
from path import Path
from sympy.utilities.codegen import codegen
from triflow.misc.misc import (cd, extract_parameters, order_field,
                               write_codegen)
from triflow.models.boundaries import analyse_boundary
from triflow.path_project import fmodel_dir

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


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
        the routines and logging.informations needed to compute the boundaries.

    """

    working_dir = fmodel_dir / folder
    logging.info(working_dir)
    U = np.load(working_dir / 'U_symb.npy')
    parameters = np.load(working_dir / 'pars_symb.npy')
    fields_order = np.load(working_dir / 'field_order.npy')
    bdc_fsymbols = np.load(working_dir / 'bdc_fsymbols.npy')
    bdc_parameters = np.load(working_dir / 'bdc_parameters.npy')

    logging.info("paramètres (ordonnés): " +
                 ' - '.join([par.name for par in parameters]))
    spec = impu.spec_from_file_location("F",
                                        working_dir /
                                        'F.cpython-36m-x86_64'
                                        '-linux-gnu.so')
    func_i = impu.module_from_spec(spec).f

    spec = impu.spec_from_file_location(
        "J", working_dir / 'J.cpython-36m-x86_64-linux-gnu.so')
    jacob_i = impu.module_from_spec(spec).j

    help_routines = {file.namebase.split('_')[1]: file.namebase for
                     file in
                     working_dir.files()
                     if (file.ext == '.f90' and
                         file.namebase.split('_')[0] == 'H' and
                         len(file.namebase.split('_')) == 2)}

    helps_i = {}
    for key, H in help_routines.items():
        spec = impu.spec_from_file_location(
            H, working_dir / '%s.cpython-36m-x86_64-linux-gnu.so' % H)
        help_i = getattr(impu.module_from_spec(spec), H.lower())
        helps_i[key] = help_i

    Fbdc_routines = sorted([file.namebase for
                            file in
                            working_dir.files()
                            if (file.ext == '.f90' and
                                file.namebase[:5] == 'F_bdc')])
    Fbounds = []
    for Fbdc in Fbdc_routines:
        try:
            spec = impu.spec_from_file_location(
                Fbdc, working_dir /
                '%s.cpython-36m-x86_64-linux-gnu.so' % Fbdc)
            Fbound = impu.module_from_spec(spec).f_bdc
        except ImportError:
            spec = impu.spec_from_file_location(
                Fbdc, working_dir / '%s.so' % Fbdc)
            Fbound = impu.module_from_spec(spec).fbdc
        Fbounds.append(Fbound)

    Jbdc_routines = sorted([file.namebase for
                            file in
                            working_dir.files()
                            if (file.ext == '.f90' and
                                file.namebase[:5] == 'J_bdc')])
    Jbounds = []
    for Jbdc in Jbdc_routines:
        try:
            spec = impu.spec_from_file_location(
                Jbdc, working_dir /
                '%s.cpython-36m-x86_64-linux-gnu.so' % Jbdc)
            Jbound = impu.module_from_spec(spec).j_bdc
        except ImportError:
            spec = impu.spec_from_file_location(Jbdc, working_dir /
                                                '%s.so' % Jbdc)
            Jbound = impu.module_from_spec(spec).j_bdc
        Jbounds.append(Jbound)

    Hsbounds = {}
    Hsbdc_routines = {key: [] for key in help_routines}
    for file in sorted(
            filter(
                lambda file: (file.ext == '.f90' and
                              file.namebase.split('_')[0] == 'H' and
                              len(file.namebase.split('_')) == 4),
                working_dir.files()),
            key=lambda file: file.namebase.split('_')[-1]):
        Hsbdc_routines[file.namebase.split('_')[1]].append(file.namebase)

    logging.debug(Hsbdc_routines)

    for key, Hbdc_routines in Hsbdc_routines.items():
        Hbounds = []
        logging.debug(Hbdc_routines)
        for Hbdc in Hbdc_routines:
            spec = impu.spec_from_file_location(
                Hbdc, working_dir /
                '%s.cpython-36m-x86_64-linux-gnu.so' % Hbdc)
            Hbound = getattr(impu.module_from_spec(spec),
                             '_'.join(Hbdc.split('_')[:-1]).lower())
            Hbounds.append(Hbound)
        Hsbounds[key] = Hbounds

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

    with cd(working_dir), open(os.devnull, 'w') as fnull:
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

    logging.info('extraction des paramètres')
    Fparams = extract_parameters(F, U)
    Jparams = extract_parameters(J, U)

    # On fait l'union des deux séries de paramètres (au cas où), et on s'assure
    # de garder la liste dans un ordre cohérent.
    parameters = sorted(list(Fparams.union(Jparams)),
                        key=lambda symbol: symbol.name)
    logging.info("paramètres (ordonnés): " +
                 ' - '.join([par.name for par in parameters]))
    logging.info("écriture et compilation des routines Fortran")
    logging.info("membre de gauche")
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

    helps_i = {}
    for key, H in Helps.items():
        H_symb = sp.Symbol('Hout')
        help_i = codegen(("H_%s" % key, sp.Eq(H_symb,
                                              H)),
                         'F95',
                         argument_sequence=(U.flatten().tolist() +
                                            parameters +
                                            [H_symb]))
        helps_i[key] = help_i

    Fbounds = []
    for i, Fbdc in enumerate(Fbdcs):
        Fbdc_matsymb = sp.MatrixSymbol('Fbdcout', *Fbdc.shape)
        Fbound = codegen(('F_bdc', sp.Eq(Fbdc_matsymb,
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
        Jbound = codegen(('J_bdc', sp.Eq(Jbdc_matsymb,
                                         Jbdc)),
                         'F95',
                         argument_sequence=(list(map(sp.Symbol,
                                                     bdc_fsymbols.flatten()
                                                     .tolist() +
                                                     bdc_parameters)) +
                                            [Jbdc_matsymb])
                         )
        Jbounds.append(Jbound)

    Hsbounds = {}
    for Hkey, Hvalue in Hsbdcs.items():
        Hbounds = []
        for j, Hbdc in enumerate(Hvalue):
            Hbdc_symb = sp.Symbol('Hbdcout')
            Hbound = codegen(('H_%s_bdc' % Hkey, sp.Eq(Hbdc_symb,
                                                       Hbdc)),
                             'F95',
                             argument_sequence=(list(map(sp.Symbol,
                                                         bdc_fsymbols.flatten()
                                                         .tolist() +
                                                         bdc_parameters)) +
                                                [Hbdc_symb])
                             )
            Hbounds.append(Hbound)
        Hsbounds[Hkey] = Hbounds

    working_dir = fmodel_dir / folder
    working_dir.rmtree_p()
    working_dir.makedirs()

    codewritter = map(lambda x: write_codegen(x, working_dir),
                      [func_i, jacob_i])

    for key, help_i in helps_i.items():
        it.chain(codewritter,
                 write_codegen(help_i,
                               working_dir))
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
    for key, Hbounds in Hsbounds.items():
        for j, Hbound in enumerate(Hbounds):
            it.chain(codewritter,
                     write_codegen(Hbound,
                                   working_dir,
                                   template=lambda filename: "%s_%i.%s" %
                                   ('.'.join(filename.split('.')[:-1]),
                                    j,
                                    filename.split('.')[-1])))
    list(codewritter)

    np.save(working_dir / 'U_symb', U)
    np.save(working_dir / 'pars_symb', parameters)
    np.save(working_dir / 'field_order', fields_order)
    np.save(working_dir / 'bdc_fsymbols', bdc_fsymbols)
    np.save(working_dir / 'bdc_parameters', bdc_parameters)

    compile_routines_fortran(folder)
