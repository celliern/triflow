#!/usr/bin/env python
# coding=utf8

import json

import numpy as np
import pytest
import xarray as xr
import yaml
from path import tempdir

from triflow import Container, Model, Simulation, retrieve_container


@pytest.fixture
def heat_model():
    model = Model(differential_equations="k * dxxT",
                  dependent_variables="T",
                  parameters="k", compiler="numpy")
    return model


@pytest.fixture
def fields(heat_model):
    x, dx = np.linspace(0, 10, 50, retstep=True, endpoint=False)
    T = np.cos(x * 2 * np.pi / 10)
    initial_fields = heat_model.fields_template(x=x, T=T)

    return initial_fields


@pytest.fixture
def simul(heat_model, fields):
    parameters = dict(periodic=True, k=1)
    simul = Simulation(heat_model, fields.copy(), parameters,
                       dt=.5, tmax=2, tol=1E-1,
                       id="test_triflow_containers")
    return simul


def test_containers_coerce(simul, fields):
    with tempdir() as container_path:
        simul.parameters["test_bool"] = True
        simul.parameters["test_list"] = []
        simul.parameters["test_object"] = type("TestObject", (object, ),
                                               dict(a=[], b={}))
        simul.attach_container(container_path)
        simul.run()


def test_containers_last(simul, fields):
    with pytest.raises(ValueError):
        simul.attach_container(None, save="")
    with tempdir() as container_path:
        simul.attach_container(container_path, save="last")
        simul.run()
        assert simul.container.data.t.size == 1
        assert simul.container.data == simul.fields


def test_containers_attached_memory(simul, fields):
    simul.attach_container(None)
    simul.run()

    assert simul.container.data.isel(t=0) == fields
    assert simul.container.data.isel(t=-1) == simul.fields
    assert simul.container.metadata == simul.parameters


def test_containers_attached_ondisk(simul, fields):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        simul.run()

        assert simul.container.data.isel(t=0) == fields
        assert simul.container.data.isel(t=-1) == simul.fields
        assert simul.container.metadata == simul.parameters

        with open(container_path / simul.id / "metadata.yml") as metafile:
            assert yaml.load(metafile) == simul.parameters
        assert (xr.open_dataset(container_path / simul.id / "data.nc") ==
                simul.container.data)
        with pytest.raises(FileExistsError):
            Container(path=container_path / simul.id, force=False, mode="w")
        Container(path=container_path / simul.id, force=True, mode="w")
        (container_path / simul.id).rmtree()
        with pytest.raises(FileNotFoundError):
            Container(path=container_path / simul.id, mode="r")


@pytest.mark.parametrize("lazy", (True, False))
def test_containers_retrieve_all(simul, lazy):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        simul.run()
        container = retrieve_container(container_path / simul.id)
        assert container.data == simul.container.data
        assert container.metadata == simul.container.metadata
        container = retrieve_container(container_path / simul.id, lazy=lazy,
                                       isel='all')
        assert container.data == simul.container.data
        assert container.metadata == simul.container.metadata


@pytest.mark.parametrize("lazy", (True, False))
def test_containers_retrieve_last(simul, lazy):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        simul.run()
        container = retrieve_container(container_path / simul.id, lazy=lazy,
                                       isel="last")
        assert container.data == simul.container.data.isel(t=-1)
        assert container.metadata == simul.container.metadata


@pytest.mark.parametrize("lazy", (True, False))
def test_containers_retrieve_dict(simul, lazy):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        simul.run()
        container = retrieve_container(container_path / simul.id,
                                       lazy=lazy, isel=dict(x=0, t=-1))
        assert container.data == simul.container.data.isel(x=0, t=-1)
        assert container.metadata == simul.container.metadata


@pytest.mark.parametrize("lazy", (True, False))
def test_containers_retrieve_list(simul, lazy):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        simul.run()
        container = retrieve_container(container_path / simul.id,
                                       lazy=lazy, isel=[0, 1, 2])
        assert container.data == simul.container.data.isel(t=[0, 1, 2])
        assert container.metadata == simul.container.metadata


@pytest.mark.parametrize("lazy", (True, False))
def test_containers_retrieve_incomplete(simul, lazy):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        next(simul)
        simul.container.flush()
        next(simul)
        simul.container.flush()
        container = retrieve_container(container_path / simul.id)
        assert container.data == simul.container.data
        assert container.metadata == simul.container.metadata


def test_containers_retrieve_backcompat(simul):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        simul.run()
        Container.get_all(container_path / simul.id)
        Container.get_last(container_path / simul.id)
        with open(container_path / simul.id / "metadata.yml", "r") as f:
            pars = yaml.load(f)
        with open(container_path / simul.id / "Treant.16486.json", "w") as f:
            json.dump(dict(categories=pars), f)
        (container_path / simul.id / "metadata.yml").remove()
        Container.get_all(container_path / simul.id)


@pytest.mark.parametrize("lazy", (True, False))
def test_containers_merge(simul, lazy):
    with tempdir() as container_path:
        simul.attach_container(container_path)
        next(simul)
        simul.container.flush()
        next(simul)
        simul.container.flush()
        sliced_data = simul.container.data.load().copy()
        Container.merge_datafiles(container_path / simul.id)
        with pytest.raises(FileExistsError):
            Container.merge_datafiles(container_path / simul.id)
        fields = xr.open_dataset(container_path / simul.id / "data.nc")
        assert fields == sliced_data


@pytest.mark.parametrize("mode", ("w", "a"))
def test_containers_meta_set(mode):
    cont = Container(None, mode)
    cont.metadata = dict(test="foo")
    assert cont.metadata["test"] == "foo"

    with tempdir() as container_path:
        cont = Container(container_path / "test_meta", mode)
        cont.metadata = dict(test="foo")

        with open(container_path / "test_meta" / "metadata.yml", "r") as f:
            pars = yaml.load(f)
        assert cont.metadata["test"] == "foo"
        assert cont.metadata["test"] == pars["test"]
        cont = Container(container_path / "test_meta", "r")
        cont.metadata = dict(test="foo")
        (container_path / "test_meta" / "metadata.yml").remove()
        assert cont.metadata is None
        assert cont.data is None

    cont = Container(None, "r")

    cont.metadata["test"] = "foo"


def test_containers_repr():
    cont = Container(None)
    str(cont)
