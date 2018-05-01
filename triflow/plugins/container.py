#!/usr/bin/env python
# coding=utf8

import json
import logging
import warnings
from collections import deque, namedtuple
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
    def __init__(self, path=None, mode='a', *, save="all",
                 metadata={}, force=False, nbuffer=50):
        self._nbuffer = nbuffer
        self._mode = mode
        self._metadata = metadata
        self.save = save
        self._cached_data = deque([], self._n_save)
        self._collector = None
        self.path = path = Path(path).abspath() if path else None

        if not path:
            return

        if self._mode == "w" and force:
            path.rmtree_p()

        if self._mode == "w" and not force and path.exists():
            raise FileExistsError(
                "Directory %s exists, set force=True to override it" % path)

        if self._mode == "r" and not path.exists():
            raise FileNotFoundError("Container not found.")
        path.makedirs_p()

        with open(self.path / 'metadata.yml', 'w') as yaml_file:
            yaml.dump(self._metadata,
                      yaml_file, default_flow_style=False)

    @property
    def save(self):
        return "last" if self._n_save else "all"

    @save.setter
    def save(self, value):
        if value == "all":
            self._n_save = None
        elif value == "last" or value == -1:
            self._n_save = 1
        else:
            raise ValueError('save argument accept only "all", "last" or -1 '
                             'as value, not %s' % value)

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
        def get_t_fields(simul):
            return simul.t, simul.fields

        def expand_fields(inps):
            return self._expand_fields(*inps)

        def get_last(list_fields):
            return list_fields[-1]

        accumulation_stream = (stream
                               .map(get_t_fields)
                               .map(expand_fields))

        self._collector = collect(accumulation_stream)
        if self.save == "all":
            self._collector.map(self._concat_fields).sink(self._write)
        else:
            self._collector.map(get_last).sink(self._write)

        (accumulation_stream
         .partition(self._nbuffer)
         .sink(self._collector.flush))

        return self._collector

    def flush(self):
        if self._collector:
            self._collector.flush()

    def _write(self, concatenated_fields):
        if concatenated_fields is not None and self.path:
            target_file = self.path / "data_%i.nc" % uuid1()
            concatenated_fields.to_netcdf(target_file)
            self._cached_data = deque([], self._n_save)
            if self.save == "last":
                [file.remove()
                 for file in self.path.glob("data_*.nc")
                 if file != target_file]

    def __repr__(self):
        repr = """path:   {path}
{data}""".format(path=self.path, data=self.data)
        return repr

    def __del__(self):
        self.flush()

    @property
    def data(self):
        try:
            if self.path:
                return open_mfdataset(self.path / "data*.nc")
            return self._concat_fields(self._cached_data)
        except OSError:
            return

    @property
    def metadata(self):
        try:
            if self.path:
                with open(self.path / 'metadata.yml', 'r') as yaml_file:
                    return yaml.load(yaml_file)
            return self._metadata
        except OSError:
            return

    @metadata.setter
    def metadata(self, parameters):
        if self._mode == "r":
            return
        for key, value in parameters.items():
            self._metadata[key] = value
        if self.path:
            with open(self.path / 'info.yml', 'w') as yaml_file:
                yaml.dump(self._metadata,
                          yaml_file, default_flow_style=False)

    @staticmethod
    def retrieve(path, isel='all', lazy=True):
        path = Path(path)
        try:
            data = open_dataset(path / "data.nc")
            lazy = True
        except FileNotFoundError:
            data = open_mfdataset(path / "data*.nc",
                                  concat_dim="t").sortby("t")
        try:
            with open(Path(path) / 'metadata.yml', 'r') as yaml_file:
                metadata = yaml.load(yaml_file)
        except FileNotFoundError:
            # Ensure retro-compatibility with older version
            with open(path.glob("Treant.*.json")[0]) as f:
                metadata = json.load(f)["categories"]

        if isel == 'last':
            data = data.isel(t=-1)
        elif isel == 'all':
            pass
        elif isinstance(isel, dict):
            data = data.isel(**isel)
        else:
            data = data.isel(t=isel)

        if not lazy:
            return FieldsData(data=data.load(),
                              metadata=AttrDict(**metadata))

        return FieldsData(data=data,
                          metadata=AttrDict(**metadata))

    @staticmethod
    def get_last(path):
        warnings.warn(
            "get_last method is deprecied and will be removed in a "
            "future version. Please use retrieve(path, 'last') instead.",
            DeprecationWarning
        )

        return TriflowContainer.retrieve(path, isel=[-1], lazy=False)

    @staticmethod
    def get_all(path):
        warnings.warn(
            "get_last method is deprecied and will be removed in a "
            "future version. Please use retrieve(path) instead.",
            DeprecationWarning
        )

        return TriflowContainer.retrieve(path, isel="all", lazy=False)

    def merge(self, override=True):
        if self.path:
            return TriflowContainer.merge_datafiles(self.path,
                                                    override=override)

    @staticmethod
    def merge_datafiles(path, override=False):
        path = Path(path)

        if (path / "data.nc").exists() and not override:
            raise FileExistsError(path / "data.nc")
        (path / "data.nc").remove_p()

        split_data = open_mfdataset(path / "data*.nc",
                                    concat_dim="t").sortby("t")
        split_data.to_netcdf(path / "data.nc")
        merged_data = open_dataset(path / "data.nc", chunks=split_data.chunks)

        if not split_data.equals(merged_data):
            (path / "data.nc").remove()
            raise IOError("Unable to merge data ")

        [file.remove() for file in path.files("data_*.nc")]
        return path / "data.nc"
