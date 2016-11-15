#!/usr/bin/env python3
# coding=utf8

import importlib.util as impu
import itertools
import logging
import os
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from functools import partial, reduce
from logging import debug, info
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sps
import sympy as sp
from numpy import f2py
from path import Path, getcwdu
from scipy.signal import periodogram
from sympy.utilities.autowrap import autowrap
from sympy.utilities.codegen import codegen

from triflow.boundaries import analyse_boundary
from triflow.model_2fields import model as model2
from triflow.model_4fields import model as model4
from triflow.model_full_fourrier import model as modelfull
from triflow.path_project import fmodel_dir


@contextmanager
def cd(dirname):
    """

    Parameters
    ----------
    dirname :
        

    Returns
    -------

    
    """
    try:
        Path(dirname)
        curdir = Path(getcwdu())
        dirname.chdir()
        yield
    finally:
        curdir.chdir()


def write_codegen(code, working_dir,
                  template=lambda filename: "%s" % filename):
    """

    Parameters
    ----------
    code :
        param working_dir:
    template :
        Default value = lambda filename: "%s" % filename)
    working_dir :
        

    Returns
    -------

    
    """
    for file in code:
        info("write %s" % template(file[0]))
        with open(working_dir / template(file[0]), 'w') as f:
            f.write(file[1])


def extract_parameters(M, U):
    """Permet de trouver les paramètres, cad les symboles qui ne sont
    pas contenus dans le champs de solution U.

    Parameters
    ----------
    M :
        param U:
    U :
        

    Returns
    -------

    
    """
    parameters = M.atoms(sp.Symbol).difference(set(U.flatten()))
    return parameters


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


def make_routines_fortran(model, boundary):
    """Permet de génerer les fonctions binaires directement utilisable par la
    classe Solver. La fonction en entrée doit générer les vecteurs symboliques
    U (avec les variables discrètes), F et J respectivement le membre de droite
    et le jacobien du modèle.

    Parameters
    ----------
    model :
        param boundary:
    boundary :
        

    Returns
    -------

    
    """
    U, F, J, pars, Helps = model()
    fields_order = order_field(U)

    window_range, nvar = U.shape

    Fbdcs, Jbdcs, Hsbdcs, Ubdcs = boundary((U, F, J, pars, Helps))

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

    U0 = autowrap(Ubdcs, args=(map(sp.Symbol, bdc_fsymbols.flatten().tolist() +
                                   bdc_parameters)))

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
                                                     helps_i, U0,
                                                     fields_order,
                                                     bdc_fsymbols,
                                                     bdc_parameters)


def load_routines_fortran(folder):
    """Si un modèle est déjà sauvegardé, il est possible de le charger sous
    une forme accepté par le solver via cette fonction.

    Parameters
    ----------
    folder :
        

    Returns
    -------

    
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

    spec = impu.spec_from_file_location("F",
                                        working_dir /
                                        'F.cpython-35m-x86_64-linux-gnu.so')
    func_i = impu.module_from_spec(spec).f

    spec = impu.spec_from_file_location(
        "J", working_dir / 'J.cpython-35m-x86_64-linux-gnu.so')
    jacob_i = impu.module_from_spec(spec).j

    spec = impu.spec_from_file_location("Ubdc",
                                        working_dir /
                                        'Ubdc.cpython-35m-x86_64-linux-gnu.so')
    U0 = impu.module_from_spec(spec).ubdc

    help_routines = [file.namebase for
                     file in
                     working_dir.files()
                     if (file.ext == '.f90' and
                         file.namebase[0] == 'H' and
                         not file.namebase[:4] == 'Hbdc')]
    helps_i = []
    for H in help_routines:
        spec = impu.spec_from_file_location(
            H, working_dir / '%s.cpython-35m-x86_64-linux-gnu.so' % H)
        help_i = impu.module_from_spec(spec).h
        helps_i.append(help_i)

    Fbdc_routines = sorted([file.namebase for
                            file in
                            working_dir.files()
                            if (file.ext == '.f90' and
                                file.namebase[:4] == 'Fbdc')])
    Fbounds = []
    for Fbdc in Fbdc_routines:
        spec = impu.spec_from_file_location(
            Fbdc, working_dir / '%s.cpython-35m-x86_64-linux-gnu.so' % Fbdc)
        Fbound = impu.module_from_spec(spec).fbdc
        Fbounds.append(Fbound)

    Jbdc_routines = sorted([file.namebase for
                            file in
                            working_dir.files()
                            if (file.ext == '.f90' and
                                file.namebase[:4] == 'Jbdc')])
    Jbounds = []
    for Jbdc in Jbdc_routines:
        spec = impu.spec_from_file_location(
            Jbdc, working_dir / '%s.cpython-35m-x86_64-linux-gnu.so' % Jbdc)
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
            spec = impu.spec_from_file_location(
                Hbdc, working_dir /
                '%s.cpython-35m-x86_64-linux-gnu.so' % Hbdc)
            Hbound = impu.module_from_spec(spec).hbdc
            Hbounds.append(Hbound)
        Hsbounds.append(Hbounds)

    return func_i, jacob_i, U, parameters, helps_i, (Fbounds, Jbounds,
                                                     Hsbounds, U0,
                                                     fields_order,
                                                     bdc_fsymbols,
                                                     bdc_parameters)


def comp_function(routine, working_dir=Path('.')):
    """

    Parameters
    ----------
    routine :
        param working_dir:  (Default value = Path('.')
    working_dir :
        (Default value = Path('.')

    Returns
    -------

    
    """
    fnull = open(os.devnull, 'w')
    with cd(working_dir):
        subprocess.call(["f2py", "-c", "-m",
                         "%s" % routine, "%s.f90" % routine],
                        stdout=fnull)


def compile_routines_fortran(folder):
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


def cache_routines_fortran(model, boundary, folder):
    """

    Parameters
    ----------
    model :
        param boundary:
    folder :
        
    boundary :
        

    Returns
    -------

    
    """
    U, F, J, pars, Helps = model()
    fields_order = order_field(U)

    window_range, nvar = U.shape

    Fbdcs, Jbdcs, Hsbdcs, Ubdc = boundary((U, F, J, pars, Helps))

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

    Ubdc_matsymb = sp.MatrixSymbol('Ubdcout', *Ubdc.shape)
    Ubound = codegen(('Ubdc', sp.Eq(Ubdc_matsymb,
                                    Ubdc)),
                     'F95',
                     argument_sequence=(list(map(sp.Symbol,
                                                 bdc_fsymbols.flatten()
                                                 .tolist() +
                                                 bdc_parameters)) +
                                        [Ubdc_matsymb])
                     )

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
                      [func_i, jacob_i, Ubound])

    for i, help_i in enumerate(helps_i):
        itertools.chain(codewritter,
                        write_codegen(help_i,
                                      working_dir,
                                      template=lambda filename: "%s_%i.%s" %
                                      ('.'.join(filename.split('.')[:-1]),
                                       i,
                                       filename.split('.')[-1])))
    for i, Fbound in enumerate(Fbounds):
        itertools.chain(codewritter,
                        write_codegen(Fbound,
                                      working_dir,
                                      template=lambda filename: "%s_%i.%s" %
                                      ('.'.join(filename.split('.')[:-1]),
                                       i,
                                       filename.split('.')[-1])))
    for i, Jbound in enumerate(Jbounds):
        itertools.chain(codewritter,
                        write_codegen(Jbound,
                                      working_dir,
                                      template=lambda filename: "%s_%i.%s" %
                                      ('.'.join(filename.split('.')[:-1]),
                                       i,
                                       filename.split('.')[-1])))
    for i, Hbounds in enumerate(Hsbounds):
        for j, Hbound in enumerate(Hbounds):
            itertools.chain(codewritter,
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


class Solver(object):

    """docstring for ClassName"""

    def __init__(self, routine_gen):
        (self.func_i, self.jacob_i,
         self.U, self.parameters,
         self.helpers,
         ((self.Fbdc, self.Jbdc, self.Hsbdc, self.U0,
           self.fields_order,
           self.bdc_fsymbols,
           self.bdc_parameters))) = routine_gen()
        self.window_range, self.nvar = self.U.shape

    def check_pars(self, pars, parlist):
        """

        Parameters
        ----------
        pars :
            param parlist:
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
        nvar = self.nvar
        fields = []
        for ivar in range(nvar):
            fields.append(flat_data[ivar::nvar].squeeze())
        return fields

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
            param **pars:
        **pars :
            

        Returns
        -------

        
        """
        nvar = self.nvar
        window_range = self.window_range
        bdc_range = int((window_range - 1) / 2)
        Nx = int(data.size / nvar)
        Fpars = self.check_pars(pars, self.parameters)
        # Boundary
        # U = boundary.U(data, t)
        F = np.zeros(data.shape)
        Ui = np.zeros([nvar * window_range])
        for i in range(bdc_range, Nx - bdc_range):
            Ui[:] = data[(i - bdc_range) * nvar: (i + bdc_range + 1) * nvar]
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
            param **pars:
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
            for i in range(bdc_range + 1, Nx - bdc_range - 1):
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
                             in self.Hsbdc[j]]).flatten()

            H[:(bdc_range)] = Hbdc[:(bdc_range)]
            H[-(bdc_range):] = Hbdc[(bdc_range):]
            Hs.append(H)
        return Hs

    def compute_U0(self, data, **pars):
        """

        Parameters
        ----------
        data :
            param **pars:
        **pars :
            

        Returns
        -------

        
        """
        nvar = self.nvar
        window_range = self.window_range
        bdc_range = int((window_range - 1) / 2)
        Nx = int(data.size / nvar)
        bdcpars = self.check_pars(pars, self.bdc_parameters)
        bdc_args = (data[:(bdc_range + 2) * nvar].tolist() +
                    data[-(bdc_range + 2) * nvar:].tolist() +
                    bdcpars)
        U0 = self.U0(*bdc_args).squeeze()
        return U0

    def compute_J(self, data, **pars):
        """

        Parameters
        ----------
        data :
            param **pars:
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
        for i in range(bdc_range + 1, Nx - bdc_range - 1):
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
        J[-bdc_range * nvar:, -nvar * (bdc_range + 2):] = Jbdctopright

        return J

    def compute_J_sparse(self, data, **pars):
        """

        Parameters
        ----------
        data :
            param **pars:
        **pars :
            

        Returns
        -------

        
        """
        nvar = self.nvar
        window_range = self.window_range
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
                (data, (row_padded, col_padded)), shape=(nvar * Nx, nvar * Nx))
            data, t = yield J_sparse

    def start_simulation(self, U0, t0, **pars):
        """

        Parameters
        ----------
        U0 :
            param t0:
        t0 :
            
        **pars :
            

        Returns
        -------

        
        """
        return Simulation(self, U0, t0, **pars)


class Simulation(object):
    """ """

    def __init__(self, solver, U0, t0, **pars):
        self.solver = solver
        self.pars = pars
        self.nvar = self.solver.nvar
        self.U = U0
        self.t = t0
        self.iterator = self.compute()

    def check_stability_fft(self, theta, dt, U, F, tol_stability=1E-8):
        """

        Parameters
        ----------
        theta :
            param dt:
        U :
            param F:
        tol_stability :
            Default value = 1E-8)
        dt :
            
        F :
            

        Returns
        -------

        
        """
        stable = False
        fields = self.get_fields(U)
        freq, ampls = periodogram(fields)
        for field in self.get_fields(time_deriv):
            stable *= np.mean(field) < tol_stability
        return stable

    def writter(self, t, U):
        """

        Parameters
        ----------
        t :
            param U:
        U :
            

        Returns
        -------

        
        """
        pass

    def driver(self):
        """Like a hook, called after every successful step:
        this is what is returned to the user after each iteration. Can be
        easily replaced to an other driver, for example in order
        to manage the time step.

        Parameters
        ----------

        Returns
        -------

        
        """
        pass

    def display(self):
        """Like a hook, called after every successful step:
        this is what is returned to the user after each iteration. Can be
        easily replaced to an other driver.

        Parameters
        ----------

        Returns
        -------

        
        """
        return self.solver.get_fields(self.U), self.t

    def takewhile(self, U):
        """

        Parameters
        ----------
        U :
            

        Returns
        -------

        
        """
        return True

    # def FE_scheme(self):
    #     while True:
    #         self.F = F = self.solver.compute_F(self.U,
    #                                            self.solver.boundary,
    #                                            self.t,
    #                                            **self.pars)
    #         U = self.U + self.pars['dt'] * F
    #         yield U

    def FE_scheme(self):
        """ """
        from scipy.integrate import ode
        solv = ode(lambda t, x: self.solver.compute_F(x,
                                                      **self.pars))
        x = np.linspace(0, self.pars['Nx'] * self.pars['dx'], self.pars['Nx'])
        solv.set_integrator('dopri5')
        solv.set_initial_value(self.U)
        while solv.successful:
            U = solv.integrate(self.t + self.pars['dt'])
            U[1::self.nvar] = (U[1::self.nvar] *
                               (-(np.tanh((x - self.pars['Nx'] * .9) /
                                          (self.pars['dx'] *
                                           self.pars['Nx'] / 40)) + 1) /
                                2 + 1) +
                               ((np.tanh((x - self.pars['Nx'] * .9) /
                                         (self.pars['dx'] *
                                          self.pars['Nx'] / 40)) + 1) / 2) *
                               self.pars['qini'])
            self.t += self.pars['dt']
            self.pars['t'] = self.t
            yield U

    def BE_scheme(self):
        """ """
        Jcomp = self.solver.compute_J_sparse(self.U,
                                             **self.pars)
        next(Jcomp)
        while True:
            self.F = F = self.solver.compute_F(self.U,
                                               **self.pars)
            J = Jcomp.send((self.U, self.t))
            B = self.pars['dt'] * (F -
                                   self.pars['theta'] * J @ self.U) + self.U
            J = (sps.identity(self.nvar * self.pars['Nx'],
                              format='csc') -
                 self.pars['theta'] * self.pars['dt'] * J)
            U = sps.linalg.splu(J).solve(B)
            yield U

    def compute(self):
        """ """
        nvar = self.nvar
        self.pars['Nx'] = int(self.U.size / nvar)

        if self.pars['theta'] != 0:
            numerical_scheme = self.BE_scheme()
        else:
            numerical_scheme = self.FE_scheme()

        for self.i, self.U in enumerate(itertools.takewhile(self.takewhile,
                                                            numerical_scheme)):
            self.U[:nvar] = self.solver.compute_U0(self.U, **self.pars)
            self.writter(self.t, self.U)
            self.t += self.pars['dt']
            self.pars['t'] = self.t
            self.driver()
            yield self.display()

    def __iter__(self):
        return self.iterator

    def __next__(self):
        return next(self.iterator)


if __name__ == '__main__':
    from triflow.boundaries import periodic_boundary, openflow2_boundary
    import multiprocessing as mp
    logger = logging.getLogger()
    logger.handlers = []
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    processes = []
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model2, openflow2_boundary,
                                      '2fields_open')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model4, openflow2_boundary,
                                      '4fields_open')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model2, periodic_boundary,
                                      '2fields_per')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model4, periodic_boundary,
                                      '4fields_per')))
    # for n in range(3, 11):
    #     processes.append(mp.Process(target=cache_routines_fortran,
    #                                 args=(lambda: modelfull(n),
    #                                       periodic_boundary,
    #                                       'full_fourrier%i_per' % n)))
    #     processes.append(mp.Process(target=cache_routines_fortran,
    #                                 args=(lambda: modelfull(n),
    #                                       openflow_boundary,
    #                                       'full_fourrier%i_open' % n)))

    for process in processes:
        process.start()

    finish = False
    while not finish:
        finish = True
        for process in processes:
            finish *= not process.is_alive()
    print("processes are finish")

    for process in processes:
        process.join()
