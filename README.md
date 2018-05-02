# Triflow

[![Build Status](https://travis-ci.org/locie/triflow.svg?branch=master)](https://travis-ci.org/locie/triflow)
[![Coverage Status](https://coveralls.io/repos/github/locie/triflow/badge.svg?branch=dev)](https://coveralls.io/github/locie/triflow?branch=master)

[![Build Status](https://travis-ci.org/locie/triflow.svg?branch=dev)](https://travis-ci.org/locie/triflow)
[![Coverage Status](https://coveralls.io/repos/github/locie/triflow/badge.svg?branch=dev)](https://coveralls.io/github/locie/triflow?branch=dev)

## Installation

### External requirements

This library is written for python &gt;= 3.5.

The library is based on Theano, thus extra dependencies like fortran and
C compiler are needed, see Theano install page for extra information:

<http://deeplearning.net/software/theano/install.html>

On v0.5.0, it is possible to choose between theano and numpy (which provide similar features). numpy will be slower but with no compilation time, which is handy for testing and prototyping.

### via PyPI

``` {.sourceCode .bash}
pip install triflow
```

will install the package and

``` {.sourceCode .bash}
pip install triflow --upgrade
```

will update an old version of the library.

use sudo if needed, and the user flag if you want to install it without
the root privileges:

``` {.sourceCode .bash}
pip install --user triflow
```

### via github

You can install the last version of the library using pip and the github
repository:

``` {.sourceCode .bash}
pip install git+git://github.com/locie/triflow.git
```

## Introduction

### Motivation

The aim of this library is to have a (relatively) easy way to write
transient dynamic systems with 1D finite difference discretization, with
fast temporal solvers.

The main two parts of the library are:

- symbolic tools defining the spatial discretization, with boundary
    taking into account in a separated part
- a fast temporal solver written in order to use the sparsity of the
    finite difference method to reduce the memory and CPU usage during
    the solving

Moreover, extra tools are provided and the library is written in a
modular way, allowing an easy extension of these different parts (see
the plug-in module of the library.)

The library fits well with an interactive usage (in a jupyter notebook).
The dependency list is actually larger, but on-going work target a
reduction of the stack complexity.

### Model writing

All the models are written as function generating the F vector and the
Jacobian matrix of the model defined as `dtU = F(U)`.

The symbolic model is written as a simple mathematic equation. For
example, a diffusion advection model can be written as:

``` {.sourceCode .python}
from triflow import Model

equation_diff = "k * dxxU - c * dxU"
dependent_var = "U"
physical_parameters = ["k", "c"]

model = Model(equation_diff, dependent_var, physical_parameters)
```

### Example

``` {.sourceCode .python}
import numpy as np
import pylab as pl
from triflow import Model, Simulation

model = Model("k * dxxU - c * dxU",
              "U", ["k", "c"])

x, dx = np.linspace(0, 1, 200, retstep=True)
U = np.cos(2 * np.pi * x * 5)
fields = model.fields_template(x=x, U=U)

parameters = dict(c=.03, k=.001, dx=dx, periodic=False)

t = 0
dt = 5E-1
tmax = 2.5

pl.plot(fields.x, fields.U, label=f't: {t:g}')


def dirichlet_condition(t, fields, pars):
    fields.U[0] = 1
    fields.U[-1] = 0
    return fields, pars


simul = Simulation(model, t, fields, parameters, dt,
                   hook=dirichlet_condition, tmax=tmax)

for i, (t, fields) in enumerate(simul):
    print(f"iteration: {i}\t",
          f"t: {t:g}", end='\r')
    pl.plot(fields.x, fields.U, label=f't: {t:g}')

pl.xlim(0, 1)
legend = pl.legend(loc='best')

pl.show()
```

### NEWS

v0.4.7:

- adding tensor flow support with full testing
- adding post-processing in bokeh fields display

v0.4.12:

- give user choice of compiler
  - get out tensorflow compiler (not really efficient, lot of maintenance trouble)
  - give access to theano and numpy compiler
- upwind scheme support
- using xarray as fields backend, allowing easy post process and save
- update display and containers
- adding repr string to all major classes

v0.5.0:

- WARNING: some part of the API has changed:
  - Simulation signature has changed. `t` arg is now optional (with t=0) as default and `physical_parameters` is now `parameters`.
  - The displays have been completely rewritten, and the previous API is depreciated. Users are encouraged to modify their scripts or to stick to the ^0.4 triflow versions.
- move schemes from plugins to core
- compilers: remove tensorflow, add numpy which is way slower but has no compilation overhead.
- displays and containers are connected to the simulation via `streamz`
- add post-processing.
- real-time display is now based on [Holoviews](https://holoviews.org/). Backward compatibility for display is broken and users are encouraged to modify their scripts or to stick to the ^0.4 triflow versions.
- use poetry to manage dependencies.
- use `tqdm` to display simulation update.

### ROADMAP / TODO LIST

The following items are linked to a better use of solid external libs:

- change all the display and container workflow:
  - use streamz to allow pipeline like way to add display / probing / post-process :done:
  - use holoviews as main way to do real-time plotting :done:
  - use xarray multi netcdf files to reduce IO lack of performance :done:
- better use of external solving lib:
  - merge triflow.plugins.schemes and scipy.integrate.OdeSolver API :todo:
  - use scipy.integrate.solve_ivp for triflow temporal scheme solving (making it more robust) :todo:
  - main goal is to have better two-way integration with scipy :todo:

These are linked to the triflow core

- build a robust boundary condition API :todo:
- work on dimension extension, allowing 2D resolution and more :todo:
- allow auxiliary function to make some complex model easier to write :todo:
- allow a choice on the finite difference scheme, on a global way or term by term :todo:
- test and propose other compilers (Cython, numba, pythran?) :todo:
- work on adaptive spatial and temporal mesh :todo:

These are far away but can be very interesting:

- implement continuation algorithm working with triflow (separate project?)
- try other kind of discretization scheme (separate project each?)
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
triflow language, the different spatial discretization and so on...

### License

This project is licensed under the term of the [MIT license](LICENSE)

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.584101.svg)](https://doi.org/10.5281/zenodo.584101)
