#!/usr/bin/env python
# coding=utf8
from contextlib import contextmanager
from path import Path, getcwdu
from logging import debug, error, info
import sympy as sp


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


def coroutine(func):
    def wrapper(*arg, **kwargs):
        generator = func(*arg, **kwargs)
        next(generator)
        return generator
    return wrapper


def write_codegen(code, working_dir,
                  template=lambda filename: "%s" % filename):
    """

    Parameters
    ----------
    code :

    working_dir :

    template :
         (Default value = lambda filename: "%s" % filename)

    Returns
    -------

    """

    for file in code:
        info("write %s" % template(file[0]))
        with open(working_dir / template(file[0]), 'w') as f:
            f.write(file[1])


def extract_parameters(M, U):
    """

    Parameters
    ----------
    M :

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


def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]


def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]
