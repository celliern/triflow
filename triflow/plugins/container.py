#!/usr/bin/env python
# coding=utf8

from collections import namedtuple
import logging
import time

import datreant.core as dtr
import numpy as np
from datreant.data import attach  # noqa: used when imported
from path import Path
from queue import Queue

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())

FieldsData = namedtuple("FieldsData", ["data", "metadata"])


class TreantContainer(object):

    def __init__(self, path, mode='a', *,
                 t0=0, initial_fields=None, metadata={},
                 nbuffer=50, timeout=180):
        self._nbuffer = nbuffer
        self._timeout = timeout
        self.keys = None
        self._mode = mode
        self._writing_queue = None

        path = Path(path)

        if self._mode == "w":
            Path(path).rmtree_p()

        if self._mode == "r":
            if not path.exists():
                raise IOError("Container not found.")
            return

        self.treant = dtr.Treant(path)

        for key, value in metadata.items():
            self.treant.categories[key] = value
        initial_keys = self.treant.data.keys()

        if self._mode == "a" and len(initial_keys) != 0:
            self.keys = initial_keys
            self.keys.remove('x')
            self.keys.remove('t')

        if initial_fields:
            if len(initial_keys) == 0:
                self._init_data(t0, initial_fields)
                return
            raise ValueError("Trying to initialize with fields"
                             " and non-empty datasets")

    def _init_data(self, t0, initial_fields):
        if self._mode == "r":
            return
        logging.debug('init data')
        self.keys = list(initial_fields.keys()).copy()
        self.keys.remove('x')
        self.treant.data.add('x', initial_fields["x"])
        try:
            self.keys.remove('y')
            self.treant.data.add('y', initial_fields["y"])
        except ValueError:
            pass

        for key in self.keys:
            self.treant.data.add(key,
                                 initial_fields[key]
                                 .reshape((1,
                                           *initial_fields[key].shape)))
        self.treant.data.add('t', np.array(t0))
        self._writing_queue = Queue(self._nbuffer)
        self._time_last_flush = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self._mode == "r":
            return
        self.flush()
        self.close()

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
        full_t = []
        full_fields = {key: [] for key in self.keys}
        while not self._writing_queue.empty():
            t, fields = self._writing_queue.get()
            logging.debug('get')
            full_t.append(t)
            for key in self.keys:
                full_fields[key].append(fields[key]
                                        .reshape((1,
                                                  *fields[key].shape)))
        full_fields = {key: np.vstack(value)
                       for key, value
                       in full_fields.items()}
        self.treant.data['t'] = np.append(self.treant.data['t'],
                                          np.array(full_t))

        for key in self.keys:
            self.treant.data[key] = np.append(
                self.treant.data[key],
                full_fields[key],
                axis=0
            )
        log.debug("Queue empty.")

    def append(self, t, fields):
        if self._mode == "r":
            return
        if not self.keys:
            self._init_data(t, fields)
            return
        self._writing_queue.put((t, fields))
        # time.sleep(0.01)
        if (self._writing_queue.full() or
                (time.time() - self._time_last_flush > self._timeout)):
            logging.debug('Flushing')
            self.flush()
            self._time_last_flush = time.time()

    def set(self, t, fields):
        self._init_data(t, fields)

    @property
    def data(self):
        return self.treant.data

    @property
    def metadata(self):
        return self.treant.categories

    @metadata.setter
    def metadata(self, parameters):
        if self._mode == "r":
            return
        for key, value in parameters.items():
            self.treant.categories[key] = value

    def __getitem__(self, key):
        try:
            data = self.treant.data[key]
            if self._mode == 'r':
                data.setflags(write=False)
            return data
        except KeyError:
            return self.metadata[key]

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError

    @staticmethod
    def get_last(path):
        with TreantContainer(path, "r") as container:
            return FieldsData(data=dict(**container.data),
                              metadata=dict(**container.metadata))
