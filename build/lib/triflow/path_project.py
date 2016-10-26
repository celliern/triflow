#!/usr/bin/env python3

import sys

from path import Path as p
from appdirs import AppDirs
import triflow

dirs = AppDirs('triflow', 'celliern')

conf_dir = p(dirs.user_config_dir)
data_dir = p(dirs.user_data_dir)
install_dir = p(triflow.__path__[0])
fmodel_dir = data_dir / "fmodel"
conf_dir.mkdir_p()
data_dir.mkdir_p()
fmodel_dir.mkdir_p()

sys.path.append(fmodel_dir)
