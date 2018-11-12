#!/usr/bin/env python
# coding=utf-8

from .utils import enable_notebook, tqdm  # noqa
from .core import temporal_schemes as schemes  # noqa
from .core.temporal_schemes import Stationnary as StationnarySolver
from .core.model import Model  # noqa
from .core.simulation import Simulation  # noqa
from .core.system import PDESys, PDEquation  # noqa
from .core.grid_builder import GridBuilder  # noqa
from .core import compilers

from .plugins.container import TriflowContainer as Container  # noqa
from .plugins.displays import TriflowDisplay as Display  # noqa

import logging
from logging import NullHandler

log = logging.getLogger(__name__)
log.handlers = []
log.addHandler(logging.NullHandler())

retrieve_container = Container.retrieve
display_fields = Display.display_fields
display_probe = Display.display_probe
