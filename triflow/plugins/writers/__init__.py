#!/usr/bin/env python
# coding=utf8
import logging

from triflow.plugins.writers.bokeh import bokeh_nb_writer
from triflow.plugins.writers.datreant import (datreant_step_writer,
                                              datreant_steps_writer)
from triflow.plugins.writers.remote import (remote_step_writer,
                                            remote_steps_writer)

try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
