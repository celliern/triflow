ROADMAP / TODO LIST
-------------------

The following items are linked to a better use of solid external libs:

- change all the display and container workflow:
  - use streamz to allow pipeline like way to add display / probing / post-process
  - use holoviews as main way to do real-time plotting
  - use xarray multi netcdf files to reduce IO lack of performance
- better use of external solving lib:
  - merge triflow.plugins.schemes and scipy.integrate.OdeSolver API
  - use scipy.integrate.solve_ivp for triflow temporal scheme solving (making it more robust)
  - main goal is to have better two-way integration with scipy

These are linked to the triflow core

- build a robust boundary condition API
- work on dimension extension, allowing 2D resolution and more
- allow auxiliary function to make some complex model easier to write
- allow a choice on the finite difference scheme, on a global way or term by term
- test and propose other compilers (Cython, numba, pythran?)
- work on adaptive spatial and temporal mesh

These are far away but can be very interesting:

- implement continuation algorithm working with triflow (separate project?)
- try other kind of discretisation scheme (separate project each?)
  - Finite volume
  - Finite element?

The final (and very ambitious goal) is to provide a robust framework allowing
to link the mathematical language (and a natural way to write the model,
as natural as possible) with a high performance and robust solving of
the numerical system. There is a trade-off between the ease of use and the
performance, for this software, all mandatory dependencies have to be
pip-installable (as opposite to dedalus project or fenics project), even if
some non-mandatory dependencies can be harder to install.

If we go that further, it may be interesting to split the project with the
triflow language, the different spatial discretisation and so on...
