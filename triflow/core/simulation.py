#!/usr/bin/env python
# coding=utf8

import itertools as it
import logging
from uuid import uuid1

from ..plugins import schemes

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Simulation(object):
    """ """

    def __init__(self, model, fields, t, pars, id=None,
                 hook=lambda fields, t, pars: (fields, pars)):
        self.id = str(uuid1()) if id is None else id
        self.model = model
        self.pars = pars
        self.nvar = fields.size

        self.fields = fields
        self.t = t
        self.i = 0
        self.iterator = it.takewhile(self.takewhile, self.compute())
        self.drivers = []
        self.writers = []
        self.scheme = schemes.RODASPR(model)
        self.signals = {}
        self.status = 'created'
        self.history = None
        self.hook = hook

    def compute(self):
        while True:
            self.fields, self.pars = self.hook(self.fields, self.t, self.pars)
            self.fields, self.t = self.scheme(self.fields,
                                              self.t, self.pars['dt'],
                                              self.pars, hook=self.hook)
            yield self.fields, self.t

    def compute_until_finished(self):
        logging.info('simulation %s computing until the end' % self.id)
        self.pars['tmax']
        for iteration in self.iterator:
            logging.info('simulation reached time %.2f, iteration %i' %
                         (self.t, self.i))
        self.status = 'over'
        return iteration

    def add_signal(self, field, signal):
        self.signals[field] = signal

    def add_driver(self, driver):
        self.drivers.append(driver)

    def set_scheme(self, scheme):
        self.scheme = scheme(self.model)

    def driver(self, t):
        """
        Modify the parameters at each internal time steps. The driver have
        to be appened to the attribute drivers.
        Parameters
        ----------
        t : actual time


        Returns
        -------
        None

        """

        for driver in self.drivers:
            driver(self, t)

        for field, signal in self.signals.items():
            self.pars['%sini' % field] = signal(t)

    def takewhile(self, outputs):
        """

        Parameters
        ----------
        U :


        Returns
        -------
        Stopping condition for the simulation: without overide
        this will raise an error if the film thickness go less than 0 and
        exit when tmax is reached if in the parameters.
        """

        if self.pars.get('tmax', None) is None:
            return True
        if self.t > self.pars.get('tmax', None):
            self.status = 'finished'
            return False
        return True
        return True

    def __iter__(self):
        return self.iterator

    def __next__(self):
        return next(self.iterator)
