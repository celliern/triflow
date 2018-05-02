#!/usr/bin/env python
# coding=utf8

from functools import partial

import numpy as np
from scipy.sparse import csc_matrix
from sympy import lambdify


def theano_compiler(model):
    """Take a triflow model and return optimized theano routines.

    Parameters
    ----------
    model: triflow.Model:
        Model to compile

    Returns
    -------
    (theano function, theano_function):
        Optimized routine that compute the evolution equations and their
        jacobian matrix.
    """
    from theano import tensor as T
    from theano.ifelse import ifelse
    import theano.sparse as ths
    from theano import function

    def th_Min(a, b):
        if isinstance(a, T.TensorVariable) or isinstance(b, T.TensorVariable):
            return T.where(a < b, a, b)
        return min(a, b)

    def th_Max(a, b):
        if isinstance(a, T.TensorVariable) or isinstance(b, T.TensorVariable):
            return T.where(a < b, b, a)
        return max(a, b)

    def th_Heaviside(a):
        if isinstance(a, T.TensorVariable):
            return T.where(a < 0, 1, 1)
        return 0 if a < 0 else 1

    mapargs = {arg: T.vector(arg)
               for arg, sarg
               in zip(model._args, model._symbolic_args)}

    to_feed = mapargs.copy()

    x_th = mapargs['x']
    N = x_th.size
    L = x_th[-1] - x_th[0]
    dx = L / (N - 1)
    to_feed['dx'] = dx

    periodic = T.scalar("periodic", dtype="int32")

    middle_point = int((model._window_range - 1) / 2)

    th_args = [mapargs[key]
               for key
               in [*model._indep_vars,
                   *model._dep_vars,
                   *model._help_funcs,
                   *model._pars]] + [periodic]

    map_extended = {}

    for (varname, discretisation_tree) in \
            model._symb_vars_with_spatial_diff_order.items():
        pad_left, pad_right = model._bounds

        th_arg = mapargs[varname]

        per_extended_var = T.concatenate([th_arg[pad_left:],
                                          th_arg,
                                          th_arg[:pad_right]])
        edge_extended_var = T.concatenate([[th_arg[0]] * middle_point,
                                           th_arg,
                                           [th_arg[-1]] * middle_point])

        extended_var = ifelse(periodic,
                              per_extended_var,
                              edge_extended_var)

        map_extended[varname] = extended_var
        for order in range(pad_left, pad_right + 1):
            if order != 0:
                var = ("{}_{}{}").format(varname,
                                         'm' if order < 0 else 'p',
                                         np.abs(order))
            else:
                var = varname
            new_var = extended_var[order - pad_left:
                                   extended_var.size +
                                   order - pad_right]
            to_feed[var] = new_var

    F = lambdify((model._symbolic_args),
                 expr=model.F_array.tolist(),
                 modules=[T, {"Max": th_Max,
                              "Min": th_Min,
                              "Heaviside": th_Heaviside}])(
        *[to_feed[key]
          for key
          in model._args]
    )

    F = T.concatenate(F, axis=0).reshape((model._nvar, N)).T
    F = T.stack(F).flatten()

    J = lambdify((model._symbolic_args),
                 expr=model.J_array.tolist(),
                 modules=[T, {"Max": th_Max,
                              "Min": th_Min,
                              "Heaviside": th_Heaviside}])(
        *[to_feed[key]
          for key
          in model._args]
    )

    J = [j if j != 0 else T.constant(0.)
         for j in J]
    J = [j if not isinstance(j, (int, float)) else T.constant(j)
         for j in J]
    J = T.stack([T.repeat(j, N) if j.ndim == 0 else j
                 for j in J])
    J = J[model._sparse_indices[0]].T.squeeze()

    i = T.arange(N).dimshuffle([0, 'x'])
    idx = T.arange(N * model._nvar).reshape((N, model._nvar)).T
    edge_extended_idx = T.concatenate([T.repeat(idx[:, :1],
                                                middle_point,
                                                axis=1),
                                       idx,
                                       T.repeat(idx[:, -1:],
                                                middle_point,
                                                axis=1)],
                                      axis=1).T.flatten()
    per_extended_idx = T.concatenate([idx[:, -middle_point:],
                                      idx,
                                      idx[:, :middle_point]],
                                     axis=1).T.flatten()
    extended_idx = ifelse(periodic,
                          per_extended_idx,
                          edge_extended_idx)

    rows = T.tile(T.arange(model._nvar),
                  model._window_range * model._nvar) + i * model._nvar
    cols = T.repeat(T.arange(model._window_range * model._nvar),
                    model._nvar) + i * model._nvar
    rows = rows[:, model._sparse_indices].reshape(J.shape).flatten()
    cols = extended_idx[cols][:, model._sparse_indices] \
        .reshape(J.shape).flatten()

    permutation = T.argsort(cols)

    J = J.flatten()[permutation]
    rows = rows[permutation]
    cols = cols[permutation]
    count = T.zeros((N * model._nvar + 1,), dtype=int)
    uq, cnt = T.extra_ops.Unique(False, False, True)(cols)
    count = T.set_subtensor(count[uq + 1], cnt)

    indptr = T.cumsum(count)
    shape = T.stack([N * model._nvar, N * model._nvar])
    sparse_J = ths.CSC(J, rows, indptr, shape)
    F_theano_function = function(inputs=th_args,
                                 outputs=F,
                                 on_unused_input='ignore',
                                 allow_input_downcast=True)
    J_theano_function = function(inputs=th_args,
                                 outputs=sparse_J,
                                 on_unused_input='ignore',
                                 allow_input_downcast=True)

    return F_theano_function, J_theano_function


def numpy_compiler(model):
    """Take a triflow model and return optimized numpy routines.

    Parameters
    ----------
    model: triflow.Model:
        Model to compile

    Returns
    -------
    (numpy function, numpy function):
        Optimized routine that compute the evolution equations and their
        jacobian matrix.
    """

    def np_Min(args):
        a, b = args
        return np.where(a < b, a, b)

    def np_Max(args):
        a, b = args
        return np.where(a < b, b, a)

    def np_Heaviside(a):
        return np.where(a < 0, 1, 1)

    f_func = lambdify((model._symbolic_args),
                      expr=model.F_array.tolist(),
                      modules=[{"amax": np_Max,
                                "amin": np_Min,
                                "Heaviside": np_Heaviside},
                               "numpy"])

    j_func = lambdify((model._symbolic_args),
                      expr=model._J_sparse_array.tolist(),
                      modules=[{"amax": np_Max,
                                "amin": np_Min,
                                "Heaviside": np_Heaviside},
                               "numpy"])

    compute_F = partial(compute_F_numpy, model, f_func)
    compute_J = partial(compute_J_numpy, model, j_func)

    return compute_F, compute_J


def init_computation_numpy(model, *input_args):
    mapargs = {key: input_args[i]
               for i, key
               in enumerate([*model._indep_vars,
                             *model._dep_vars,
                             *model._help_funcs,
                             *[*model._pars, "periodic"]])}
    x = mapargs["x"]
    N = x.size
    L = x[-1] - x[0]
    dx = L / (N - 1)
    periodic = mapargs["periodic"]
    middle_point = int((model._window_range - 1) / 2)

    args = [mapargs[key]
            for key
            in [*model._indep_vars,
                *model._dep_vars,
                *model._help_funcs,
                *model._pars]] + [periodic]

    mapargs['dx'] = dx

    map_extended = mapargs.copy()

    for (varname, discretisation_tree) in \
            model._symb_vars_with_spatial_diff_order.items():
        pad_left, pad_right = model._bounds

        arg = mapargs[varname]
        if periodic:
            extended_var = np.concatenate([arg[pad_left:],
                                           arg,
                                           arg[:pad_right]])
        else:
            extended_var = np.concatenate([[arg[0]] * middle_point,
                                           arg,
                                           [arg[-1]] * middle_point])

        map_extended[varname] = extended_var
        for order in range(pad_left, pad_right + 1):
            if order != 0:
                var = ("{}_{}{}").format(varname,
                                         'm' if order < 0 else 'p',
                                         np.abs(order))
            else:
                var = varname
            new_var = extended_var[order - pad_left:
                                   extended_var.size +
                                   order - pad_right]
            map_extended[var] = new_var
    return args, map_extended, N, middle_point, periodic


def compute_F_numpy(model, f_func, *input_args):
    args, map_extended, N, middle_point, periodic = \
        init_computation_numpy(model, *input_args)
    F = f_func(*[map_extended[key]
                 for key
                 in model._args])
    F = np.concatenate(F, axis=0).reshape((model._nvar, N)).T
    F = np.stack(F).flatten()
    return F


def compute_J_numpy(model, j_func, *input_args):
    args, map_extended, N, middle_point, periodic = \
        init_computation_numpy(model, *input_args)
    J = j_func(*[map_extended[key]
                 for key
                 in model._args])

    J = np.stack([np.repeat(j, N) if len(
        np.array(j).shape) == 0 else j for j in J])
    J = J.T.squeeze()

    i = np.arange(N)[:, None]
    idx = np.arange(N * model._nvar).reshape((N, model._nvar)).T

    if periodic:
        extended_idx = np.concatenate([idx[:, -middle_point:],
                                       idx,
                                       idx[:, :middle_point]],
                                      axis=1).T.flatten()
    else:
        extended_idx = np.concatenate([np.repeat(idx[:, :1],
                                                 middle_point,
                                                 axis=1),
                                       idx,
                                       np.repeat(idx[:, -1:],
                                                 middle_point,
                                                 axis=1)],
                                      axis=1).T.flatten()

    rows = np.tile(np.arange(model._nvar),
                   model._window_range * model._nvar) + i * model._nvar
    cols = np.repeat(np.arange(model._window_range * model._nvar),
                     model._nvar) + i * model._nvar
    rows = rows[:, model._sparse_indices].reshape(J.shape)
    cols = extended_idx[cols][:, model._sparse_indices].reshape(J.shape)
    rows = rows.flatten()
    cols = cols.flatten()

    sparse_J = csc_matrix((J.flatten(), (rows, cols)),
                          shape=(N * model._nvar, N * model._nvar))
    return sparse_J
