#!/usr/bin/env python
# coding=utf8

import numpy as np


class BaseFields:
        """Specialized container which expose the data as a structured numpy array,
          give access to the dependants variables and the herlpers function as
          attributes (as a numpy rec array) and is able to give access to a flat
          view of the dependent variables only (which is needed by the ode
          solvers for all the linear algebra manipulation).

          Parameters
          ----------
          **inputs : numpy.array
              named argument providing x, the dependent variables and the helper functions. All of these are mendatory and a KeyError will be raised if a data is missing.

          Attributes
          ----------
          array : numpy.array
            vanilla numpy array containing the data
          size : int
            Number of discretisation nodes
          """  # noqa
        @staticmethod
        def factory(dependent_variables, helper_functions):
            """Fields factory generating specialized container build around a
              triflow Model.

              Parameters
              ----------
              dependent_variables : iterable of str
                  name of the dependent variables
              helper_functions : iterable of str
                  name of the helper functions

              Returns
              -------
              triflow.BaseFields
                  Specialized container which expose the data as a structured numpy array
              """  # noqa
            Field = BaseFields
            Field.dependent_variables = dependent_variables
            Field.helper_functions = helper_functions
            return Field

        def __init__(self, **inputs):
            self._keys = (['x'] +
                          list(self.dependent_variables) +
                          list(self.helper_functions))
            [self.__setattr__(key, inputs[key]) for key in set(self._keys)]

            self.size = len(self.x)

            data = list(zip(*[getattr(self, var)
                              for var in self._keys]))
            self.array = np.array(data)
            self._dtype = [(var, float) for var in self._keys]
            for var in self._keys:
                self.__setattr__(var, self.structured[var].squeeze())

        @property
        def flat(self):
            """numpy.ndarray.view: flat view of the main numpy array
            """  # noqa
            return self.array.ravel()

        @property
        def structured(self):
            """numpy.ndarray.view: structured view of the main numpy array
            """  # noqa
            return self.array.view(dtype=self._dtype)

        @property
        def uarray(self):
            """numpy.ndarray.view: view of the dependent variables of the main numpy array
            """  # noqa
            return self.array[:, 1: (1 + len(self.dependent_variables))]

        @property
        def uflat(self):
            """return a flatten **copy** of the main numpy array with only the
            dependant variables.

            Be carefull, modification of these data will not be reflected on
            the main array!
            """  # noqa
            uflat = self.array[:, 1: (1 +
                                      len(self.dependent_variables))].ravel()
            return uflat

        def fill(self, flat_array):
            """take a flat numpy array and update inplace the dependent
            variables of the container

            Parameters
            ----------
            flat_array : numpy.ndarray
                flat array which will be put in the dependant variable flat array.
            """  # noqa
            self.uarray[:] = flat_array.reshape(self.uarray.shape)

        def __getitem__(self, index):
            return self.structured[index].squeeze()

        def __iter__(self):
            return (self.array[i] for i in range(self.size))

        def copy(self):
            old_values = {var: getattr(self, var).squeeze()
                          for var in self._keys}

            return self.__class__(**old_values)

        def __repr__(self):
            return self.structured.__repr__()
