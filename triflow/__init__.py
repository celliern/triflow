#!/usr/bin/env python
# coding=utf8

from triflow.core import schemes
from triflow.core.model import Model  # noqa
from triflow.core.simulation import Simulation  # noqa
from triflow.plugins import signals, displays, container  # noqa
from triflow.plugins.container import Container  # noqa
from triflow.plugins.displays import display_1D  # noqa
from triflow.plugins.displays import display_1D as display_fields  # noqa
from triflow.plugins.displays import display_0D  # noqa
from triflow.plugins.displays import display_0D as display_probes  # noqa

import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
