#!/usr/bin/env python
# coding=utf8

import copy
import pickle

import numpy as np
import pytest
from path import Path

from triflow import Model
from triflow.core.fields import BaseFields


@pytest.fixture
def model():
    model = Model(differential_equations=["dxxU1", "dxxU2"],
                  dependent_variables=["U1", "U2"], help_functions='s')
    return model


@pytest.fixture
def fields_dict():
    x = np.linspace(0, 1, 100)
    U1 = np.cos(x)
    U2 = np.sin(x)
    s = np.sin(x)
    fields = dict(x=x, U1=U1, U2=U2, s=s)
    return fields


@pytest.fixture
def fields2D_dict():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 10)
    U1 = np.cos(x)[:, None] * y[None, :]
    U2 = np.sin(x)
    fields = dict(x=x, y=y, U1=U1, U2=U2)
    return fields


def test_fields_construct(model, fields_dict):
    fields1 = model.fields_template(**fields_dict)
    fields2 = BaseFields.factory1D(["U1", "U2"], ["s"])(**fields_dict)
    for var in ["x", "U1", "U2", "s"]:
        assert (fields1[var] == fields2[var]).all()


def test_fields_api():
    model = Model(differential_equations=["dxxU1", "dxxU2"],
                  dependent_variables=["U1", "U2"], help_functions='s')
    x = np.linspace(0, 1, 100)
    U1 = np.cos(x)
    U2 = np.sin(x)
    s = np.sin(x)
    fields = model.fields_template(x=x, U1=U1, U2=U2, s=s)
    print(fields)
    assert np.isclose(fields.uflat,
                      np.vstack([U1, U2]).flatten('F')).all()
    assert np.isclose(fields.uarray.to_array(),
                      np.vstack([U1, U2])).all()
    assert np.isclose(fields["x"], x).all()
    assert np.isclose(fields["U1"], U1).all()
    assert np.isclose(fields["U2"], U2).all()
    assert np.isclose(fields["s"], s).all()
    assert fields.size == x.size


def test_fields_reduce(fields_dict):
    fields = BaseFields.factory1D(["U1", "U2"], ["s"])(**fields_dict)
    assert fields == pickle.loads(pickle.dumps(fields))


def test_fields_copy(fields_dict):
    fields = BaseFields.factory1D(["U1", "U2"], ["s"])(**fields_dict)
    assert fields == copy.copy(fields)
    assert fields == copy.deepcopy(fields)


def test_fields_exports(fields_dict):
    fields = BaseFields.factory1D(["U1", "U2"], ["s"])(**fields_dict)
    fields.to_csv("/tmp/test_triflow.csv")
    assert Path("/tmp/test_triflow.csv").exists()


def test_fields2D_exports(fields2D_dict):
    fields = BaseFields.factory(["x", "y"],
                                [("U1", ("x", "y")),
                                 ("U2", ("x", ))], [])(**fields2D_dict)
    with pytest.raises(ValueError):
        fields.to_csv("/tmp/test_triflow.csv")
    with pytest.raises(ValueError):
        fields.to_clipboard()
