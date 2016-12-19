#!/usr/bin/env python
# coding=utf8

import numpy as np
from project_path import *
from scipy.stats import distributions as dist
from triflow.misc.materials import water
from triflow.triflow_helper.log import init_log
from triflow.triflow_helper.sampling import (generate_full_design,
                                             generate_random_design,
                                             generate_sample)

if __name__ == '__main__':
    init_log(log_dir, 'samples', 'INFO', 'INFO')

    fixed = {}
    fixed['Ct'] = 0
    fixed['Re'] = 15

    design = generate_random_design({'Bi': ([1E-3, 80], dist.lognorm(1.7))},
                                    criterion='center',
                                    nsamples=2)
    design = generate_full_design(Bi=[list(row.values())[0] for row in design],
                                  freq=np.arange(60, 101, 20))

    samples = generate_sample(water, fixed, design, '10ff_open', n=8)
    samples.to_csv(data_dir / 'samples.csv')
