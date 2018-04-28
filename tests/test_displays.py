#!/usr/bin/env python
# coding=utf8

import numpy as np
import pytest
import path

import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from triflow import Model, Simulation, display_fields, display_probe  # noqa


@pytest.fixture
def heat_model():
    model = Model(differential_equations="k * dxxT",
                  dependent_variables="T",
                  parameters="k", compiler="numpy")
    return model


@pytest.fixture
def simul(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)
    parameters = dict(periodic=True, k=1)
    simul = Simulation(heat_model, initial_fields, parameters,
                       dt=.5, tmax=2, tol=1E-1)
    return simul


def test_display_fields(simul):
    display_fields(simul,
                   on_disk="test", on_disk_folder="/tmp/triflow_test_fields")
    simul.run()


def test_display_probes(simul):
    display_probe(simul, function=lambda simul: simul.timer.total,
                  on_disk="test", on_disk_folder="/tmp/triflow_test_probes")
    simul.run()


def test_display_mul(simul):
    display_fields(simul) * display_fields(simul)
    display_fields(simul) * display_fields(simul).hv_curve


def test_display_add(simul):
    display_fields(simul) + display_fields(simul)
    display_fields(simul) + display_fields(simul).hv_curve


@pytest.mark.parametrize("fmt",
                         ["png", "svg", "pdf"])
def test_display_on_disk(simul, fmt):
    display = display_fields(simul, on_disk="test",
                             on_disk_folder="/tmp/triflow_test",
                             fmt=fmt)
    simul.run()
    [process.join() for process in display._writers]
    assert len(path.Path("/tmp/triflow_test/").glob("test_*.%s" % fmt)) == 5


def test_display_api(simul):
    display = display_fields(simul)
    display.hv_curve
