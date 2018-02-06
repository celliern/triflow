#!/usr/bin/env python
# coding=utf8

import logging
import warnings
from collections import namedtuple
from uuid import uuid1

import yaml
from path import Path
from streamz import collect
from xarray import concat, open_dataset, open_mfdataset

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())

FieldsData = namedtuple("FieldsData", ["data", "metadata"])


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def coerce_attr(key, value):
    value_type = type(value)
    if value_type in [int, float, str]:
        return value
    for cast in (int, float, str):
        try:
            value = cast(value)
            log.debug("Illegal netCDF type ({}) of attribute for {}, "
                      "casted to {}".format(value_type, key, cast))
            return value
        except TypeError:
            pass
    raise TypeError("Illegal netCDF type ({}) of attribute for {}, "
                    "auto-casting failed, tried to cast to "
                    "int, float and str")


class TriflowContainer:
    def __init__(self, path, mode='a', *,
                 metadata={}, force=False, nbuffer=50):
        self._nbuffer = nbuffer
        self._mode = mode
        self._metadata = metadata
        self._cached_data = []
        self._collector = None
        self.path = path = Path(path).abspath()

        if self._mode == "w" and force:
            path.rmtree_p()

        if self._mode == "w" and not force and path.exists():
            raise IOError("Directory %s exists, set force=True to override it"
                          % path)

        if self._mode == "r" and not path.exists():
            raise IOError("Container not found.")
        path.makedirs_p()

        with open(self.path / 'metadata.yml', 'w') as yaml_file:
            yaml.dump(self._metadata,
                      yaml_file, default_flow_style=False)

    def _expand_fields(self, t, fields):
        fields = fields.assign_coords(t=t).expand_dims("t")
        for key, value in self._metadata.items():
            fields.attrs[key] = coerce_attr(key, value)
        self._cached_data.append(fields)
        return fields

    def _concat_fields(self, fields):
        if fields:
            return concat(fields, dim="t")

    def connect(self, stream):
        accumulation_stream = (stream
                               .map(lambda simul: (simul.t, simul.fields))
                               .map(lambda inps: self._expand_fields(*inps)))

        self._collector = collect(accumulation_stream)
        self._collector.map(self._concat_fields).sink(self._write)

        (accumulation_stream
         .partition(self._nbuffer)
         .sink(self._collector.flush))

        return self._collector

    def flush(self):
        if self._collector:
            self._collector.flush()

    def _write(self, concatenated_fields):
        if concatenated_fields is not None:
            concatenated_fields.to_netcdf(self.path / "data_%i.nc" % uuid1())
            self._cached_data = []

    def __repr__(self):
        repr = """
path:   {path}
{data}
""".format(path=self.path, data=self.data)
        return repr

    def __del__(self):
        self.flush()

    @property
    def data(self):
        return open_mfdataset(self.path / "data*.nc")

    @property
    def metadata(self):
        with open(self.path / 'metadata.yml', 'r') as yaml_file:
            return yaml.load(yaml_file)

    @metadata.setter
    def metadata(self, parameters):
        if self._mode == "r":
            return
        for key, value in parameters.items():
            self._metadata[key] = value
        with open(self.path / 'info.yml', 'w') as yaml_file:
            yaml.dump(self._metadata,
                      yaml_file, default_flow_style=False)

    @staticmethod
    def retrieve(path, isel='all', lazy=True):
        path = Path(path)
        data = open_mfdataset(path / "data*.nc")

        if isel == 'last':
            data = data.isel(t=-1)
        elif isel == 'all':
            pass
        elif isinstance(isel, dict):
            data = data.isel(**isel)
        else:
            data = data.isel(t=isel)

        if not lazy:
            return data.load()

        return data

    @staticmethod
    def get_last(path):
        warnings.warn(
            "get_last method is deprecied and will be removed in a "
            "future version. Please use retrieve(path, 'last') instead.",
            DeprecationWarning
        )

        with open(Path(path) / 'metadata.yml', 'r') as yaml_file:
            metadata = yaml.load(yaml_file)
        return FieldsData(data=TriflowContainer.retrieve(path,
                                                         isel="last",
                                                         lazy=False),
                          metadata=AttrDict(**metadata))

    @staticmethod
    def get_all(path):
        warnings.warn(
            "get_last method is deprecied and will be removed in a "
            "future version. Please use retrieve(path) instead.",
            DeprecationWarning
        )

        with open(Path(path) / 'metadata.yml', 'r') as yaml_file:
            metadata = yaml.load(yaml_file)
        return FieldsData(data=TriflowContainer.retrieve(path,
                                                         isel="all",
                                                         lazy=False),
                          metadata=AttrDict(**metadata))

    @staticmethod
    def merge_datafiles(path, override=True):
        path = Path(path)

        if (path / "data.nc").exists() and not override:
            raise FileExistsError(path / "data.nc")
        (path / "data.nc").remove_p()

        split_data = open_mfdataset(path / "data*.nc")
        split_data.to_netcdf(path / "data.nc")
        merged_data = open_dataset(path / "data.nc", chunks=split_data.chunks)

        if not split_data.equals(merged_data):
            (path / "data.nc").remove()
            raise IOError("Unable to merge data ")

        [file.remove() for file in path.files("data_*.nc")]
        return path / "data.nc"
