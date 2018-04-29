Contribution guide
==================

Minor Contribution
------------------

Testing, issue reporting, new feature request are welcome (via the `github repository`_).


Make
----

- a makefile is provided, and

    + `make env` install the requirement for triflow and triflow itself
    + `make init` install the developpement requirement
    + `make clean` remove all the build artefacts
    + `make test` launch the test (doctest + pytest, with coverage)
    + `make doc` build the documentation

Testing
-------

- the master branch require 100% coverage (or good reason to ignore part of the code)
- the doctest needs to pass the test too, making sure the examples are consistants
- if the API change, the test will move to a file `deprecated.py` in tests folder, a warning will be raised and the deprecated API will be removed at the next major version (after the 1.0), or after 2 minor version (before the 1.0)

Style guide
-----------

The code is pep8 complient, and the tests is running with pytest-pep8 and pylama. the #noqa flag is allowed, but the reason has to be written.

Docstring for public API and main methods are mandatory.

Roadmap
-------

- go to continous integration via travis (or other?)
- include t as symbolic variable to allow time dependent functions (non-autonomous dynamical systems)
- test different model against analytical solutions or reference library
- benchmark the solver
- drop the 1D limitation with

    + 2D
    + ND with arbitrary independant variable
- implement various finite different scheme (as upwind), or better, an easy way to add new spatial scheme
- give the ability to use mixed scheme

.. _github repository: https://github.com/locie/triflow
