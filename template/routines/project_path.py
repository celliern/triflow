#!/usr/bin/env python
# coding=utf8

from path import Path
import inspect

working_dir = Path(inspect.getfile(inspect.currentframe())).parent.parent
figures_dir = working_dir / 'report/figures'
data_dir = working_dir / 'output_simul'
log_dir = working_dir / 'logs'
