#!/usr/bin/env python
# coding=utf-8

from .numpy_compiler import NumpyCompiler
from .theano_compiler import TheanoCompiler

def get_compiler(compiler):
    try:
        return dict(theano=TheanoCompiler, numpy=NumpyCompiler)[compiler]
    except KeyError:
        raise NotImplementedError("compiler %s not implemented yet." % compiler)