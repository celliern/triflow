#!/usr/bin/env python
# coding=utf-8

import warnings
from .numpy_compiler import NumpyCompiler

compilers = {"numpy": NumpyCompiler}

try:
    from .theano_compiler import TheanoCompiler
    compilers["theano"] = TheanoCompiler
except ImportError:
    warnings.warn("Theano cannot be imported: theano compiler will not be available.")

def get_compiler(compiler):
    try:
        return compilers[compiler]
    except KeyError:
        raise NotImplementedError("compiler %s not implemented yet." % compiler)
