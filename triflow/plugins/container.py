#!/usr/bin/env python
# coding=utf8

import logging
import time
from collections import namedtuple

import datreant.core as dtr
import numpy as np
from path import Path
from queue import Queue
from xarray import merge, open_dataset, concat

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())

FieldsData = namedtuple("FieldsData", ["data", "metadata"])


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Container(object):
    def __init__(self, path, mode='a', *,
                 t0=0, initial_fields=None,
                 metadata={}, probes=None,
                 force=False,
                 nbuffer=50, timeout=180):
        self._nbuffer = nbuffer
        self._timeout = timeout
        self._mode = mode
        self._writing_queue = None
        self._dataset = None

        path = Path(path).abspath()

        if self._mode == "w" and force:
            path.rmtree_p()

        if self._mode == "w" and not force and path.exists():
            raise IOError("Directory %s exists" % path)

        if self._mode == "r" and not path.exists():
            raise IOError("Container not found.")

        self.treant = dtr.Treant(path)
        self.path = Path(self.treant.abspath)
        self.path_data = self.path / "data.nc"

        for key, value in metadata.items():
            self.treant.categories[key] = value

        if self._mode in ["a", "r"]:
            try:
                self._dataset = open_dataset(self.path_data)
                return
            except IOError:
                raise IOError("Directory not found")

        if initial_fields and not self._dataset:
            self._init_data(t0, initial_fields, probes=probes)
            return
        else:
            raise ValueError("Trying to initialize with fields"
                             " and non-empty datasets")

    def _init_data(self, t0, initial_fields, probes=None):
        if self._mode == "r":
            return
        logging.debug('init data')
        self._dataset = initial_fields.expand_dims("t").assign_coords(t=[t0])
        self._dataset = merge([self._dataset, probes])
        for key, value in self.metadata.items():
            if isinstance(value, bool):
                value = int(value)
            self._dataset.attrs[key] = value
        self._dataset.to_netcdf(self.path_data)
        self._writing_queue = Queue(self._nbuffer)
        self._time_last_flush = time.time()

    def __repr__(self):
        repr = """
path:   {path}

{data}
""".format(path=self.path, data=self.data)
        return repr

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self._mode == "r":
            return
        self.flush()
        self.close()

    def keys(self):
        if self._dataset:
            return self._dataset.keys()
        return

    def close(self):
        if self.keys:
            self.flush()
            self._dataset.close()

    def flush(self):
        if self._mode == "r":
            return
        self._empty_queue()

    def _empty_queue(self):
        try:
            if self._writing_queue.empty():
                return
        except AttributeError:
            return
        log.debug("Flushing queue.")

        def yield_fields():
            while not self._writing_queue.empty():
                fields = self._writing_queue.get()
                yield fields

        self._dataset = concat([self._dataset, *yield_fields()], "t")
        for key, value in self.metadata.items():
            if isinstance(value, bool):
                value = int(value)
            self._dataset.attrs[key] = value
        log.debug("Queue empty.")
        log.debug("Writing.")
        self._dataset.to_netcdf(self.path_data)
        self._dataset = open_dataset(self.path_data)

    def append(self, t, fields, probes=None):
        if self._mode == "r":
            return
        if not self.keys():
            self._init_data(t, fields)
            return
        fields = fields.expand_dims("t").assign_coords(t=[t])
        fields = merge([fields, probes])
        self._writing_queue.put(fields)
        if (self._writing_queue.full() or
                (time.time() - self._time_last_flush > self._timeout)):
            logging.debug('flushing')
            self.flush()
            self._time_last_flush = time.time()

    def set(self, t, fields, probes=None, force=False):
        if self._datasets and not force:
            raise IOError("container not empty, "
                          "set force=True to override actual data")
        else:
            log.warning("container not empty, "
                        "erasing...")
            self._init_data(t, fields, probes=probes)

    @property
    def data(self):
        self.flush()
        return self._dataset

    @property
    def metadata(self):
        self.flush()
        return dict(**self.treant.categories)

    @metadata.setter
    def metadata(self, parameters):
        if self._mode == "r":
            return
        for key, value in parameters.items():
            self.treant.categories[key] = value

    def __getitem__(self, key):
        try:
            data = self._dataset[key]
            if self._mode == 'r':
                data.setflags(write=False)
            return data
        except (KeyError, TypeError):
            return self.metadata[key]

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError

    @staticmethod
    def get_all(path):
        with Container(path, "r") as container:
            return FieldsData(data=container.data,
                              metadata=AttrDict(**container.metadata))

    @staticmethod
    def get_last(path):
        with Container(path, "r") as container:
            return FieldsData(data=container.data.isel(t=[-1]),
                              metadata=AttrDict(**container.metadata))
