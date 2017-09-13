#!/usr/bin/env python
# coding=utf8

import numpy as np
import pandas as pd
from xarray import Dataset


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
        Field = BaseFields
        Field._coords = coords
        Field._dependent_variables_info = dependent_variables
        Field._helper_functions_info = helper_functions
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
        self._var_info = [*list(self._dependent_variables_info),
                          *list(self._helper_functions_info)]
        self._dependent_variables = [dep[0]
                                     for dep
                                     in self._dependent_variables_info]
        self._helper_functions = [dep[0]
                                  for dep
                                  in self._helper_functions_info]
        self._keys, self._coords_info = zip(*self._var_info)

        super().__init__(data_vars={key: (coords, inputs[key])
                                    for key, coords in self._var_info},
                         coords={coord: inputs[coord]
                                 for coord in self._coords})

    def copy(self):
        Field = BaseFields.factory(self._coords,
                                   self._dependent_variables_info,
                                   self._helper_functions_info)
        new_array = Field(**{key: self[key].values
                             for key
                             in (self._keys + self._coords)})
        return new_array

    @property
    def size(self):
        """numpy.ndarray.view: view of the dependent variables of the main numpy array
        """  # noqa
        return self["x"].size

    @property
    def uarray(self):
        """numpy.ndarray.view: view of the dependent variables of the main numpy array
        """  # noqa
        return self[self._dependent_variables]

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
                                            for c in self._coords]]
                          for key, coords in self._dependent_variables_info]
        return np.hstack(aligned_arrays).flatten("F")

    def keys(self):
        return self._keys

    def to_df(self):
        if len(self.coords) > 1:
            raise ValueError("CSV files only available for 1D arrays")
        data = {key: self.to_dict()['data_vars'][key]['data']
                for key
                in self.keys()}
        df = pd.DataFrame(dict(x=self.x, **data))
        return df

    def fill(self, uflat):
        rarray = uflat.reshape((self.coords["x"].size, -1), order="F")
        ptr = 0
        for var, coords in self._dependent_variables_info:
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
