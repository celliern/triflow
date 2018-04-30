#!/usr/bin/env python
# coding=utf8

from copy import copy, deepcopy

import numpy as np
import pandas as pd
from xarray import Dataset


def reduce_fields(coords,
                  dependent_variables,
                  helper_functions,
                  data):
    Field = BaseFields.factory(coords,
                               dependent_variables,
                               helper_functions)
    return Field(**data)


class BaseFields(Dataset):
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
    def factory(coords,
                dependent_variables,
                helper_functions):
        """Fields factory generating specialized container build around a
          triflow Model and xarray.

          Parameters
          ----------
          coords: iterable of str:
              coordinates name. First coordinate have to be shared with all
              variables
          dependent_variables : iterable tuple (name, coords)
              coordinates and name of the dependent variables
          helper_functions : iterable tuple (name, coords)
              coordinates and name of the helpers functions

          Returns
          -------
          triflow.BaseFields
              Specialized container which expose the data as a structured
              numpy array
          """
        Field = type('Field', BaseFields.__bases__,
                     dict(BaseFields.__dict__))
        Field._coords = coords
        Field.dependent_variables_info = dependent_variables
        Field.helper_functions_info = helper_functions
        Field._var_info = [*list(Field.dependent_variables_info),
                           *list(Field.helper_functions_info)]
        Field.dependent_variables = [dep[0]
                                     for dep
                                     in Field.dependent_variables_info]
        Field.helper_functions = [dep[0]
                                  for dep
                                  in Field.helper_functions_info]
        Field._keys, Field._coords_info = zip(*Field._var_info)
        return Field

    @staticmethod
    def factory1D(dependent_variables,
                  helper_functions):
        """Fields factory generating specialized container build around a
          triflow Model and xarray.
          Wrapper for 1D data.

          Parameters
          ----------
          dependent_variables : iterable for str
              name of the dependent variables
          helper_functions : iterable of str
              name of the helpers functions

          Returns
          -------
          triflow.BaseFields
              Specialized container which expose the data as a structured
              numpy array
          """
        return BaseFields.factory(("x", ),
                                  [(name, ("x", ))
                                   for name
                                   in dependent_variables],
                                  [(name, ("x", ))
                                   for name
                                   in helper_functions],)

    def __init__(self, **inputs):
        Dataset.__init__(self,
                         data_vars={key: (coords, inputs[key])
                                    for key, coords in self._var_info},
                         coords={coord: inputs[coord]
                                 for coord in self._coords})

    def __reduce__(self):
        return (reduce_fields,
                (self._coords,
                 self.dependent_variables_info,
                 self.helper_functions_info,
                 {key: self[key]
                  for key in self.keys()}))

    def copy(self, deep=True):
        new_dataset = Dataset.copy(self, deep)
        new_dataset.__dict__.update({key: (deepcopy(value)
                                           if deep
                                           else copy(value))
                                     for key, value
                                     in self.__dict__.items()})
        return new_dataset

    def __copy__(self, deep=True):
        return self.copy(deep)

    @property
    def size(self):
        """numpy.ndarray.view: view of the dependent variables of the main numpy array
        """  # noqa
        return self["x"].size

    @property
    def uarray(self):
        """numpy.ndarray.view: view of the dependent variables of the main numpy array
        """  # noqa
        return self[self.dependent_variables]

    @property
    def uflat(self):
        """return a flatten **copy** of the main numpy array with only the
        dependant variables.

        Be carefull, modification of these data will not be reflected on
        the main array!
        """  # noqa
        aligned_arrays = [self[key].values[[(slice(None)
                                             if c in coords
                                             else None)
                                            for c in self._coords]].T
                          for key, coords in self.dependent_variables_info]
        return np.vstack(aligned_arrays).flatten("F")

    def keys(self):
        return [*self._coords, *self._keys]

    def to_df(self):
        if len(self.coords) > 1:
            raise ValueError("CSV files only available for 1D arrays")
        data = {key: self[key]
                for key
                in self._keys}
        df = pd.DataFrame(dict(**data), index=self[self._coords[0]])
        return df

    def fill(self, uflat):
        rarray = uflat.reshape((self[self._coords[0]].size,
                                -1))
        ptr = 0
        for var, coords in self.dependent_variables_info:
            coords = list(coords)
            coords.remove(self._coords[0])
            next_ptr = ptr + sum([self.coords[key].size for key in coords])
            next_ptr = ptr + 1 if next_ptr == ptr else next_ptr
            self[var][:] = rarray[:, ptr:next_ptr].squeeze()
            ptr = next_ptr

    def to_csv(self, path):
        self.to_df().to_csv(path)

    def to_clipboard(self):
        self.to_df().to_clipboard()
