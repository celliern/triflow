#!/usr/bin/env python3
# coding=utf8

import importlib.util as impu
import itertools as it
import logging
import os
import subprocess
from collections import deque
from contextlib import contextmanager
from functools import partial
from logging import debug, error, info
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sps
import sympy as sp
from path import Path, getcwdu
from scipy.fftpack import irfft, rfft
from scipy.linalg import lu_factor, lu_solve, norm
from scipy.integrate import ode
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.signal import periodogram
from sympy.utilities.autowrap import autowrap
from sympy.utilities.codegen import codegen
import pylab as pl
import seaborn as sns

from triflow.boundaries import analyse_boundary
from triflow.path_project import fmodel_dir


@contextmanager
def cd(dirname):
    try:
        Path(dirname)
        curdir = Path(getcwdu())
        dirname.chdir()
        yield
    finally:
        curdir.chdir()


def write_codegen(code, working_dir,
                  template=lambda filename: "%s" % filename):
    for file in code:
        info("write %s" % template(file[0]))
        with open(working_dir / template(file[0]), 'w') as f:
            f.write(file[1])


def extract_parameters(M, U):
    """
    Permet de trouver les paramètres, cad les symboles qui ne sont
    pas contenus dans le champs de solution U.
    """
    parameters = M.atoms(sp.Symbol).difference(set(U.flatten()))
    return parameters


def order_field(U):
    order_field = list(map(lambda y:
                           next(map(lambda x:
                                    str(x).split('_')[0],
                                    y)
                                ),
                           U.T)
                       )
    return order_field


def make_routines_fortran(model, boundary):
    """
    Permet de génerer les fonctions binaires directement utilisable par la
    classe Solver. La fonction en entrée doit générer les vecteurs symboliques
    U (avec les variables discrètes), F et J respectivement le membre de droite
    et le jacobien du modèle.
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


def load_routines_fortran(folder):
    """
    Si un modèle est déjà sauvegardé, il est possible de le charger sous
    une forme accepté par le solver via cette fonction.
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
                                                     Hsbounds,
                                                     fields_order,
                                                     bdc_fsymbols,
                                                     bdc_parameters)


def comp_function(routine, working_dir=Path('.')):
    fnull = open(os.devnull, 'w')
    with cd(working_dir):
        subprocess.call(["f2py", "-c", "-m",
                         "%s" % routine, "%s.f90" % routine],
                        stdout=fnull)


def compile_routines_fortran(folder):
    """
    Permet de compiler les fonctions fortran contenu dans le dossier folder,
    doit être appelé après la fonction cache_routine_fortran.
    La compilation est faite en //, afin de gagner un peu de temps sur
    les gros modèles.
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


class Solver(object):

    """docstring for ClassName"""

    def __init__(self, routine_gen):
        (self.func_i, self.jacob_i,
         self.U, self.parameters,
         self.helpers,
         ((self.Fbdc, self.Jbdc, self.Hsbdc,
           self.fields_order,
           self.bdc_fsymbols,
           self.bdc_parameters))) = routine_gen()
        self.window_range, self.nvar = self.U.shape

    def check_pars(self, pars, parlist):
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
        nvar = self.nvar
        fields = []
        for ivar in range(nvar):
            fields.append(flat_data[ivar::nvar].squeeze())
        return fields

    def flatten_fields(self, *fields):
        flat_data = np.array(fields).flatten('F')
        return flat_data

    def compute_F(self, data, **pars):
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

    def compute_J(self, data, **pars):
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

    def compute_J_sparse(self, data, **pars):
        nvar = self.nvar
        # window_range = self.window_range
        Nx = int(data.size / nvar)

        def init_sparse():
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
        return Simulation(self, U0, t0, **pars)


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
            self.internal_iter += 1
            if np.isclose(t, next_time_step):
                next_time_step += self.pars['dt']
                dt = dt_calc
                yield U
                self.internal_iter = 0

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
            raise StopIteration('h above 0')
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


if __name__ == '__main__':
    from triflow.model_2fields import model as model2
    from triflow.model_4fields import model as model4
    # from triflow.model_full_fourrier import model as modelfull
    from triflow.boundaries import periodic_boundary, openflow_boundary
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
                                args=(model2, openflow_boundary,
                                      '2fields_open')))
    processes.append(mp.Process(target=cache_routines_fortran,
                                args=(model4, openflow_boundary,
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
