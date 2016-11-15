#!/usr/bin/env python
# coding=utf8
from contextlib import contextmanager
from path import Path, getcwdu
from logging import debug, error, info
import sympy as sp


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
