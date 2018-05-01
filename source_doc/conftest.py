# content of conftest.py
import numpy
import pytest

import triflow
import path


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy


@pytest.fixture(autouse=True)
def add_triflow(doctest_namespace):
    doctest_namespace['trf'] = triflow


@pytest.fixture(autouse=True)
def add_base_triflow(doctest_namespace):
    doctest_namespace['model'] = triflow.Model("k * dxxU - c * dxU",
                                               "U", ["k", "c"])
    doctest_namespace['parameters'] = dict(c=.03, k=.001, periodic=True)
    x = numpy.linspace(0, 1, 100, endpoint=False)
    U = numpy.cos(2 * numpy.pi * x * 5)
    fields = doctest_namespace['model'].fields_template(x=x, U=U)
    doctest_namespace['initial_fields'] = fields.copy()
    doctest_namespace['fields'] = fields.copy()
    doctest_namespace['t0'] = 0
    doctest_namespace["dt"] = 1E-1
    doctest_namespace["tmax"] = 2


@pytest.fixture(autouse=True)
def add_base_schemes(doctest_namespace):
    doctest_namespace['theta'] = .5
    doctest_namespace['integrator'] = "vode"
    doctest_namespace['kwd_integrator'] = dict()


@pytest.fixture(autouse=True)
def add_plot_dir(doctest_namespace):
    doctest_namespace['plot_dir'] = path.tempdir()
