Contribution guide
==================

Minor Contribution
------------------

Testing, issue reporting, new feature request are welcome
(via the `github repository`_).


Make
----

- a makefile is provided, and

    + `make dev` install the requirement for triflow
        and triflow itself
    + `make clean` remove all the build artefacts
    + `make test` launch the test (doctest + pytest, with coverage)
    + `make doc` build the documentation
    + `make build` build the package (doc + python dist)
    + `make publish` publish the package on pypi.

Testing
-------

- the master branch require good coverage of the code
    (good reason is required to ignore part of the code)
- the doctest needs to pass the test too, making sure the
    examples are consistants.
- if the API change, the test will move to a file `deprecated.py`
    in tests folder, a warning will be raised and the deprecated API will
    be removed at the next major version (after the 1.0).
    Exception can be made for the beta (before 1.0). In that case, these API
    change have to be documented on the changelog and the news.

Style guide
-----------

The code is pep8 complient, and the tests is running with
pytest-pep8 and pylama. The #noqa flag is allowed with comment that
specify the reason of the pep8 exception.

Docstring for public API and main methods are mandatory.

.. _github repository: https://github.com/locie/triflow
