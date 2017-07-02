[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.584101.svg)](https://doi.org/10.5281/zenodo.584101)

Installation
============

External requirements
---------------------

This library is written for python &gt;= 3.6, and I recommend to install
it via [Anaconda](https://www.continuum.io/downloads) : this is a full
python distribution including a scientific-oriented IDE, the main
scientific python libraries and the Jupyter project.

The library is based on Theano, thus extra dependecies like fortran and
C compiler are needed, see Theano install page for extra informations:

<http://deeplearning.net/software/theano/install.html>

On v4.7, it is possible to choose between theano and tensorflow (which provide similar features). Tensorflow will be slower but with faster compilation. For really huge system (as hybrid 2D finite difference / pseudo-spectral), Theano couldn't provide graph computation in a reasonable time, and TensorFlow can be an interesting alternative.

In that case, tensorflow and it's python wrapper has to be install on the computer.

via PyPI
--------

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

via github
----------

You can install the last version of the library using pip and the github
repository:

``` {.sourceCode .bash}
pip install git+git://github.com/locie/triflow.git
```

Introduction
============

Motivation
----------

The aim of this library is to have a (relatively) easy way to write
transient dynamic systems with 1D finite difference discretisation, with
fast temporal solvers.

The main two parts of the library are:

-   symbolic tools defining the spatial discretisation, with boundary
    taking into account in a separated part
-   a fast temporal solver written in order to use the sparsity of the
    finite difference method to reduce the memory and CPU usage during
    the solving

Moreover, extra tools are provided and the library is written in a
modular way, allowing an easy extension of these different parts (see
the plug-in module of the library.)

The library fits well with an interactive usage (in a jupyter notebook).
The dependency list is actually larger, but on-going work target a
reduction of the stack complexity.

Model writing
-------------

All the models are written as function generating the F vector and the
Jacobian matrix of the model defined as

$$\\frac{\\partial U}{\\partial t} = F(U)$$

The symbolic model is written as a simple mathematic equation. For
exemple, a diffusion advection model can be written as:

``` {.sourceCode .python}
from triflow import Model

equation_diff = "k * dxxU - c * dxU"
dependent_var = "U"
physical_parameters = ["k", "c"]

model = Model(equation_diff, dependent_var, physical_parameters)
```

Example
-------

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

NEWS
----

v4.7:
- adding tensor flow support with full testing
- adding post-processing in bokeh fields display


License
-------

This project is licensed under the term of the [BEERWARE
license](LICENSE)
