#!/usr/bin/env python
# coding=utf8

import itertools as it
import logging

from coolname import generate_slug
from triflow.plugins import schemes

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)


class Simulation(object):

    def __init__(self, model, fields, t, pars, id=None,
                 hook=lambda fields, t, pars: (fields, pars)):
        self.id = id if not id else generate_slug(2)
        self.model = model
        self.pars = pars

        self.fields = fields
        self.t = t
        self.i = 0
        self.scheme = schemes.RODASPR(model)
        self.status = 'created'
        self.hook = hook
        self.iterator = self.compute()

    def compute(self):
        fields = self.fields
        t = self.t
        pars = self.pars
        while True:
            fields, pars = self.hook(fields, t, pars)
            fields, t = self.scheme(fields, t, pars['dt'],
                                    pars, hook=self.hook)
            self.fields = fields
            self.t = t
            self.pars = pars
            yield fields, t

    def set_scheme(self, scheme, *args, **kwargs):
        self.scheme = scheme(self.model, *args, **kwargs)

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

    def __iter__(self):
        return it.takewhile(self.takewhile, self.compute())

    def __next__(self):
        return next(self.iterator)
