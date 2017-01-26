Introduction
===============

Motivation
-----------------

The aim of this library is to have a (relatively) easy way to write transient dynamic systems with 1D finite difference discretisation, with fast temporal solvers.

The main two parts of the library are :
* symbolic tools defining the spatial discretisation, with boundary taking into account in a separated part
* a fast temporal solver writted in order to use the sparsity of the finite difference method to reduce the memory and cpu usage during the solving

Moreover, extra tools are provided and the library is writed in a modular way allowing an easy extension of these different parts (see the plugins module of the library.)

The library fits well with an interactive usage (in a jupyter notebook). The dependency list is actually large, but on-going work target a reduction of the stack complexity.

Model writing
-----------------

This part will have huge rearengement in the future : do not trust the api!
For now, all the models are written as function generating the F vector and the Jacobian matrix of the model defined as

.. math::

    \frac{\partial U}{\partial t} = F(U)

The symbolic model is written using Sympy_ (the python symbolic library). The spatial scheme is written "by hand" (see the examples in the model page), but will be automated in the future.

For now, the stream-wise dimension should be discretized with the finite difference method, and its is possible to deal with additional dimension directly in the model, with pseudo-spectral or interpolation methods.

The stream-wise boundary conditions on the left and right side are written in a separate function, the other boundary conditions have to be set inside the model.

The way the model is deal with is not compatible with cross-stream finite difference discretisation (yet).

Model compilation
------------------

The model have to be compiled before being employed. The sympy library provide an easy way to automaticaly write the Fortran or C routine corresponding : the triflow library take advantage of that in order to speed-up the solving and avoid a bottleneck during the evaluation of the function F.

The Numpy_ library (the common tool to deal with high performance array manipulation) provide the tool f2py able to compile and import fortran subroutine in python making them available to the solver.

Numerical scheme, temporal solver
----------------------------------

.. _Sympy: http://www.sympy.org/en/index.html
.. _Numpy: http://www.sympy.org/en/index.html
