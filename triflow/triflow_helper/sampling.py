#!/usr/bin/env python
# coding=utf8

import functools as ft
import itertools as it
import logging
from multiprocessing import Pool

from hashids import Hashids
from pandas import DataFrame
from pyDOE import lhs
from sklearn.preprocessing import minmax_scale

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


def update_magical(physical_parameters, fixed, model,
                   salt, n_id, i, parametric):
    logging.info(i)
    hashids = Hashids(min_length=n_id, salt=salt)
    for parameter, value in parametric.items():
        physical_parameters[parameter] = value
    for parameter, value in fixed.items():
        physical_parameters[parameter] = value
    physical_parameters = dict(physical_parameters).copy()
    physical_parameters['name'] = hashids.encode(
        int.from_bytes(bytes('%i-%s' % (i, model), 'utf8'), byteorder='big'))
    physical_parameters['model'] = model
    return physical_parameters


def generate_sample(material, fixed, parametrics, model, n=1,
                    salt='salt require a long enough string to have random',
                    n_id=6):
    physical_parameters = material()

    partial = ft.partial(update_magical, physical_parameters,
                         fixed, model, salt, n_id)
    if n > 1:
        with Pool(n) as p:
            samples = p.starmap(partial,
                                enumerate(parametrics))
    else:
        samples = [partial(*arg) for arg in enumerate(parametrics)]
    df = DataFrame(data=samples,
                   index=[sample['name'] for sample in samples])
    return df


def generate_random_design(parameters_kwg,
                           criterion='center',
                           nsamples=100):

    lhd = lhs(len(parameters_kwg),
              samples=nsamples,
              criterion=criterion)
    design = {}
    for i, (parameter_name,
            (bound, dist)) in enumerate(parameters_kwg.items()):
        design[parameter_name] = minmax_scale(dist.ppf(lhd[:, i]),
                                              feature_range=bound)

    return [{key: design[key][i]
             for key in parameters_kwg}
            for i in range(nsamples)]


def generate_full_design(**parameters_kwg):
    keys = parameters_kwg.keys()
    design = it.product(*[parameters_kwg[key] for key in keys])
    design = map(lambda sample: {key: sample[i] for i, key in enumerate(keys)},
                 design)
    return list(design)
