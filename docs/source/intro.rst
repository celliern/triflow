Introduction
===============

Motivation
-----------------

The aim of this library is to have a (relatively) easy way to write transient dynamic systems with 1D finite difference discretisation, with fast temporal solvers.

The main two parts of the library are:
* symbolic tools defining the spatial discretisation, with boundary taking into account in a separated part
* a fast temporal solver written in order to use the sparsity of the finite difference method to reduce the memory and CPU usage during the solving

Moreover, extra tools are provided and the library is written in a modular way, allowing an easy extension of these different parts (see the plug-in module of the library.)

The library fits well with an interactive usage (in a jupyter notebook). The dependency list is actually larger, but on-going work target a reduction of the stack complexity.

Model writing
-----------------

This part will have a huge arrangement in the future: do not trust the ape!
For now, all the models are written as function generating the F vector and the Jacobian matrix of the model defined as

.. math::

    \frac{\partial U}{\partial t} = F(U)

The symbolic model is written using Sympy_ (the python symbolic library). The spatial scheme is written "by hand" (see the examples on the model page), but will be automated in the future.

For now, the stream-wise dimension should be discretized with the finite difference method, and it is possible to deal with additional dimension directly in the model, with pseudo-spectral or interpolation methods.

The stream-wise boundary conditions on the left and right side are written in a separate function, the other boundary conditions have to be set in the model.

The way the model is deal with is not compatible with cross-stream finite difference discretisation (yet).

Model compilation
------------------

The model has to be compiled before being employed. The sympy library provides an easy way to automatically write the Fortran or C routine corresponding: the Triflow library takes advantage of that in order to speedup the solving and avoid a bottleneck during the evaluation of the function F.

The Numpy_ library (the common tool to deal with high performance array manipulation) provides the tool f2py able to compile and import Fortran subroutine in python making them available to the solver. Further work is planned to write the routine in C to reduce the stack complexity.

The library provides the model binaries for the 1D model. The more complex ones as the full Fourier models should be compiled by the user.

Numerical scheme, temporal solver
----------------------------------

In order to provide fast and scalable temporal solver, the Jacobian use the `scipy sparse column matrix format`_ (which will reduce the memory usage, especialy for a large number of spatial nodes), and make available the SuperLU_ decomposition, a fast LU sparse matrix decomposition algorithm.

Different temporal schemes are provided in the plugins module:

* a forward Euler scheme
* a backward Euler scheme
* a :math: `\\theta` mixed scheme
* a BDF scheme
* A ROW scheme with fixed time stepping
* A ROW scheme with variable time stepping


.. _Sympy: http://www.sympy.org/en/index.html
.. _Numpy: http://www.sympy.org/en/index.html
.. _scipy sparse column matrix format: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csc_matrix.html
.. _SuperLU: http://crd-legacy.lbl.gov/~xiaoye/SuperLU/
