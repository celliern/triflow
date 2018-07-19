#!/usr/bin/env python
# coding=utf8

from .utils import enable_notebook, tqdm  # noqa
from .core import schemes  # noqa
from .core.schemes import Stationnary as StationnarySolver
from .core.model import Model  # noqa
from .core.simulation import Simulation  # noqa

from .plugins.container import TriflowContainer as Container  # noqa
from .plugins.displays import TriflowDisplay as Display  # noqa

import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

retrieve_container = Container.retrieve
display_fields = Display.display_fields
display_probe = Display.display_probe
