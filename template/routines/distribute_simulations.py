#!/usr/bin/env python
# coding=utf8

import logging

import pandas as pd
import toolz
from distributed import Client, client
from project_path import *
from triflow.triflow.plugins import signals
from triflow.triflow_helper.compute import (init_simulation, run_simulation,
                                            save_result)
from triflow.triflow_helper.log import init_log

c = Client()

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


init_log(log_dir, 'computing', 'DEBUG', 'DEBUG')

initial_parameters = {
    'hini': 1.,
    'qini': 1 / 3,
    'sini': 0,
    'thetaini': 0,
    'phiini': 0,
    'thermo_eq': True
}

noisy = signals.BrownNoise(fcut=.2, tmax=50)


def forced_signal(**parameters):
    freq = parameters['freq']
    l_factor = parameters['l_factor']
    forced = signals.ForcedSignal(signal_freq=freq * l_factor,
                                  offset=1, **parameters)
    return 'h', noisy + forced


initial_parameters.update({'T%iini' % i: 0 for i in range(10)})
numerical_parameters = {'t': 0,
                        'dt': 10,
                        'tmax': 50,
                        'tol': .5,
                        'max_iter': 1000
                        }
domain_parameters = {'L': lambda **pars: 1 * pars['l_factor'],
                     'Nx': 1000}
init_simulation = init_simulation((initial_parameters,
                                   numerical_parameters,
                                   domain_parameters),
                                  signal=forced_signal)
run_simulation = run_simulation(retry=10, down_factor=2)
save_result = save_result(data_dir / 'runs')


def pipe(sample):
    return toolz.pipe(sample, init_simulation, run_simulation)


if __name__ == '__main__':
    df = pd.read_csv(data_dir / 'samples.csv', index_col=0)
    samples_name, samples = zip(*df.iterrows())
    results = pipe(samples[0])
    # for simul in client.as_completed(results):
    #     save_result(simul.result())
