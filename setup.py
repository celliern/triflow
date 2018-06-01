#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

version = "0.5.2"

setup(
    name='triflow',
    # la version du code
    version=version,
    # Liste les packages à insérer dans la distribution
    packages=find_packages(),

    author="Nicolas CELLIER",
    author_email="contact@nicolas-cellier.net",
    description="Python library for falling-films modeling",

    license='MIT',

    # ext_modules=ext_modules,
    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,

    entry_points={},

    # Une url qui pointe vers la page officielle de votre lib
    url='http://github.com/celliern/triflow',
    tests_require=['pytest', 'pytest-cov', 'pylama',
                   'pytest-pep8'],
    install_requires=[
            'numpy',
            'scipy',
            'sympy',
            'theano',
            'streamz',
            'xarray',
            'holoviews',
            'bokeh',
            'path.py',
            'pendulum',
            ],
    download_url=('https://github.com/celliern/triflow/archive/v%s.tar.gz'
                  % version),
    # Il est d'usage de mettre quelques metadata à propos de sa lib
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers.

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha"
    ],
)
