#!/usr/bin/env python
# coding=utf8

from collections import namedtuple
import logging
import time

import datreant.core as dtr
from path import Path
from queue import Queue
from xarray import open_dataset, merge

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
                 t0=0, initial_fields=None, metadata={},
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
            except IOError:
                pass

        if initial_fields and not self._dataset:
            self._init_data(t0, initial_fields)
            return
        else:
            raise ValueError("Trying to initialize with fields"
                             " and non-empty datasets")

    def _init_data(self, t0, initial_fields):
        if self._mode == "r":
            return
        logging.debug('init data')
        self._dataset = initial_fields.expand_dims("t").assign_coords(t=[t0])
        self._dataset.attrs.update(self.metadata)
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
                t, fields = self._writing_queue.get()
                fields = fields.expand_dims("t").assign_coords(t=[t])
                yield fields
        self._dataset = merge([self._dataset, *yield_fields()])
        self._dataset.attrs.update(self.metadata)
        log.debug("Queue empty.")
        log.debug("Writing.")
        self._dataset.to_netcdf(self.path_data)
        self._dataset = open_dataset(self.path_data)

    def append(self, t, fields):
        if self._mode == "r":
            return
        if not self.keys():
            self._init_data(t, fields)
            return
        self._writing_queue.put((t, fields))
        # time.sleep(0.01)
        if (self._writing_queue.full() or
                (time.time() - self._time_last_flush > self._timeout)):
            logging.debug('Flushing')
            self.flush()
            self._time_last_flush = time.time()

    def set(self, t, fields, force=False):
        if self._datasets and not force:
            raise IOError("container not empty, "
                          "set force=True to override actual data")
        else:
            log.warning("container not empty, "
                        "erasing...")
            self._init_data(t, fields)

    @property
    def data(self):
        return self._dataset

    @property
    def metadata(self):
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
