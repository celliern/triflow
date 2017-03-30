#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='triflow',
    # la version du code
    version="0.3.0",
    # Liste les packages à insérer dans la distribution
    packages=find_packages(),

    author="Nicolas CELLIER",
    author_email="contact@nicolas-cellier.net",
    description="Python library for falling-films modeling",

    license='MIT',

    # long_description=open('README.md').read(),

    # ext_modules=ext_modules,
    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,

    entry_points={},

    # Une url qui pointe vers la page officielle de votre lib
    url='http://github.com/celliern/triflow',

    install_requires=[
            'mpmath',
            'numpy',
            'path',
            'pyparsing',
            'scipy',
            'six',
            'sympy',
            'toolz',
            'sym_dict',
            'coolname',
            'theano',
            'nose_parameterized'],
    extras_require={
        'visdom': ["visdom"],
        'bokeh': ["bokeh"],
        'datreant': ["datreant.core"],
    },
    download_url='https://github.com/celliern/triflow/archive/0.3.0.tar.gz',
    # Il est d'usage de mettre quelques metadata à propos de sa lib
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers.

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)
