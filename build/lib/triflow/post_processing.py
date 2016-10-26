#!/usr/bin/env python3
# coding=utf8

import importlib.util as impu
import logging
import os
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from functools import partial
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

from triflow.model_2fields import model as model2
from triflow.model_4fields import model as model4
from triflow.model_full_fourrier import model as modelfull
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


def extract_parameters(M, U):
    """
    Permet de trouver les paramètres, cad les symboles qui ne sont
    pas contenus dans le champs de solution U.
    """
    parameters = M.atoms(sp.Symbol).difference(set(U.flatten()))
    return parameters


def make_routines_fortran(model):
    """
    Permet de génerer les fonctions binaires directement utilisable par la
    classe Solver. La fonction en entrée doit générer les vecteurs symboliques
    U (avec les variables discrètes), F et J respectivement le membre de droite
    et le jacobien du modèle.
    """
    U, F, J, pars, Helps = model()

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
    return func_i, jacob_i, U, parameters, helps_i


def load_routines_fortran(folder):
    """
    Si un modèle est déjà sauvegardé, il est possible de le charger sous
    une forme accepté par le solver via cette fonction.
    """

    working_dir = fmodel_dir / folder
    info(working_dir)
    U = np.load(working_dir / 'U_symb.npy')
    parameters = np.load(working_dir / 'pars_symb.npy')
    info("paramètres (ordonnés): " +
         ' - '.join([par.name for par in parameters]))
    spec = impu.spec_from_file_location("F",
                                        working_dir /
                                        'F.cpython-35m-x86_64-linux-gnu.so')
    func_i = impu.module_from_spec(spec).f
    spec = impu.spec_from_file_location(
        "J", working_dir / 'J.cpython-35m-x86_64-linux-gnu.so')
    jacob_i = impu.module_from_spec(spec).j

    help_routines = [file.namebase for
                     file in
                     working_dir.files()
                     if (file.ext == '.f90' and file.namebase[0] == 'H')]

    helps_i = []
    for H in help_routines:
        spec = impu.spec_from_file_location(
            H, working_dir / '%s.cpython-35m-x86_64-linux-gnu.so' % H)
        help_i = impu.module_from_spec(spec).h
        helps_i.append(help_i)

    return func_i, jacob_i, U, parameters, helps_i


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


def cache_routines_fortran(model, folder):
    U, F, J, pars, Helps = model()

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

    working_dir = fmodel_dir / folder
    working_dir.rmtree_p()
    working_dir.makedirs()

    for file in func_i:
        with open(working_dir / file[0], 'w') as f:
            f.write(file[1])

    for file in jacob_i:
        with open(working_dir / file[0], 'w') as f:
            f.write(file[1])

    for i, help_i in enumerate(helps_i):
        for file in help_i:
            with open(working_dir / "%s%i.%s" % ('.'.join(file[0]
                                                          .split('.')[:-1]),
                                                 i,
                                                 file[0].split('.')[-1]),
                      'w') as f:
                f.write(file[1])

    np.save(working_dir / 'U_symb', U)
    np.save(working_dir / 'pars_symb', parameters)

    compile_routines_fortran(folder)


class Solver(object):

    """docstring for ClassName"""

    def __init__(self, routine_gen):
        (self.func_i, self.jacob_i,
         self.U, self.parameters,
         self.helpers) = routine_gen()
        self.window_range, self.nvar = self.U.shape

    def check_pars(self, pars):
        pars = [pars[key.name]
                for key
                in self.parameters]
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
        Nx = int(data.size / nvar)
        pars = self.check_pars(pars)
        Upadded = np.pad(data, nvar * int((window_range - 1) / 2),
                         mode='wrap')
        F = np.zeros(nvar * Nx)
        Ui = np.zeros([nvar * window_range])
        for i in range(Nx):
            Ui[:] = Upadded[i * nvar:nvar * (i + window_range)]
            Fi = self.func_i(*Ui, *pars)
            for ivar in range(nvar):
                F[i * nvar + ivar] = Fi[ivar, 0]
        return F

    def compute_F(self, data, **pars):
        nvar = self.nvar
        window_range = self.window_range
        Nx = int(data.size / nvar)
        pars = self.check_pars(pars)
        Upadded = np.pad(data, nvar * int((window_range - 1) / 2),
                         mode='wrap')
        F = np.zeros(nvar * Nx)
        Ui = np.zeros([nvar * window_range])
        for i in range(Nx):
            Ui[:] = Upadded[i * nvar:nvar * (i + window_range)]
            Fi = self.func_i(*Ui, *pars)
            for ivar in range(nvar):
                F[i * nvar + ivar] = Fi[ivar, 0]
        return F

    def compute_Hs(self, data, **pars):
        nvar = self.nvar
        window_range = self.window_range
        Nx = int(data.size / nvar)
        pars = self.check_pars(pars)
        Upadded = np.pad(data, nvar * int((window_range - 1) / 2),
                         mode='wrap')
        Hs = []
        for help_i in self.helpers:
            H = np.zeros(Nx)
            Ui = np.zeros([nvar * window_range])
            for i in range(Nx):
                Ui[:] = Upadded[i * nvar:nvar * (i + window_range)]
                Hi = help_i(*Ui, *pars)
                H[i] = Hi
            Hs.append(H)
        return Hs

    def periodic_boundary(self, J):
        nvar = self.nvar
        window_range = self.window_range
        Nx = int(J.shape[0] / nvar)
        up_range = int(np.ceil(window_range / 2))
        down_range = int(np.floor(window_range / 2))
        Jnew = np.zeros([nvar * Nx, nvar * Nx])
        Jnew[:] = J[:, down_range * nvar:-up_range * nvar]
        Jnew[:, :up_range * nvar] += J[:, -up_range * nvar:]
        Jnew[:, -down_range * nvar:] += J[:, :down_range * nvar]
        return Jnew

    def compute_J(self, data, boundary, **pars):
        nvar = self.nvar
        window_range = self.window_range
        Nx = int(data.size / nvar)
        pars = self.check_pars(pars)
        Upadded = np.pad(data, nvar * int((window_range - 1) / 2),
                         mode='wrap')
        J = np.zeros([nvar * Nx, nvar * (Nx + window_range)])
        Ui = np.zeros([nvar * window_range])
        for i in range(Nx):
            Ui[:] = Upadded[i * nvar: i * nvar + window_range * nvar]
            Ji = self.jacob_i(*Ui, *pars)
            for ivar in range(nvar):
                J[i * nvar + ivar,
                  i * nvar:nvar * (i + window_range)] = Ji[ivar]
        J = boundary(J)
        return J

    def compute_J_sparse(self, data, boundary, **pars):
        nvar = self.nvar
        window_range = self.window_range

        Nx = int(data.size / nvar)
        # debug("Nx : %i" % Nx)
        # debug("data_size : %i" % data.size)
        # debug("nvar : %i" % nvar)

        def init_sparse():
            rand_data = np.random.rand(*data.shape)
            random_J = self.compute_J(rand_data,
                                      lambda args: args, **pars)
            full_coordinates = np.indices(random_J.shape)

            Jcoordinate_array = np.zeros((2, nvar * Nx, nvar * Nx), dtype=int)

            up_range = int(np.ceil(window_range / 2))
            down_range = int(np.floor(window_range / 2))
            Jcoordinate_array[:] = full_coordinates[
                :, :, down_range * nvar:-up_range * nvar]
            Jcoordinate_array[:, -nvar * window_range:, :up_range *
                              nvar] = full_coordinates[:,
                                                       -nvar * window_range:,
                                                       -up_range * nvar:]
            Jcoordinate_array[:, :nvar * window_range,
                              -down_range * nvar:
                              ] = full_coordinates[:,
                                                   :nvar * window_range,
                                                   :down_range * nvar]

            Jpadded = tuple(Jcoordinate_array)

            maskJ = random_J[Jpadded] != 0
            row_padded = np.indices((nvar * Nx, nvar * Nx))[0][maskJ]
            col_padded = np.indices((nvar * Nx, nvar * Nx))[1][maskJ]
            row = full_coordinates[0][Jpadded][maskJ]
            col = full_coordinates[1][Jpadded][maskJ]
            return row_padded, col_padded, row, col

        row_padded, col_padded, row, col = init_sparse()
        data = yield
        while True:
            J = self.compute_J(data,
                               lambda args: args,
                               **pars)
            data = J[tuple([row, col])]
            J_sparse = sps.csc_matrix(
                (data, (row_padded, col_padded)), shape=(nvar * Nx, nvar * Nx))
            data = yield J_sparse

    # def check_stability_time_deriv(self, data, tol=1E-8, **pars):
    #     time_deriv = self.compute_F(data, **pars)
    #     stable = False
    #     for field in self.get_fields(time_deriv):
    #         stable *= np.mean(field) < tol
    #     return stable

    # def check_stability_fft(self, data, tol=1E-8):
    #     stable = False
    #     fields = self.get_fields(data)
    #     freq, ampls = periodogram(fields)
    #     for field in self.get_fields(time_deriv):
    #         stable *= np.mean(field) < tol
    #     return stable

    def compute(self, U0, t0, bmagic=True, **pars):
        nvar = self.nvar
        Nx = int(U0.size / nvar)
        U = U0
        t = t0
        Jcomp = self.compute_J_sparse(U, self.periodic_boundary, **pars)
        next(Jcomp)
        theta, dt = yield
        while True:
            if theta != 0:
                if bmagic:
                    F = self.compute_F(U, **pars)
                    if bmagic:
                        J = Jcomp.send(U)
                        B = dt * (F - theta * J @ U) + U
                        J = sps.identity(nvar * Nx,
                                         format='csc') - theta * dt * J
                        U = sps.linalg.splu(J).solve(B)
                    else:
                        J = self.compute_J(U, self.periodic_boundary, **pars)
                        B = dt * (F - theta * J @ U) + U
                        J = np.identity(nvar * Nx) - theta * dt * J
                        U = solve(J, B)
            else:
                F = self.compute_F(U, **pars)
                U += dt * F
                t += dt
            t += dt
            next_iter_par = yield U, t
            theta, dt = next_iter_par if next_iter_par else theta, dt

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = []
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    cache_routines_fortran(model2, '2fields')
    cache_routines_fortran(model4, '4fields')
    for n in range(3, 11):
        cache_routines_fortran(lambda: modelfull(n), 'full_fourrier%i' % n)

    # cache_routines_fortran(lambda: modelfull(3), 'full_fourrier%i' % 3)
