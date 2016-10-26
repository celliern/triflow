#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from numpy.distutils.core import setup

setup(
    name='triflow',
    # la version du code
    version="0.1.1",
    # Liste les packages à insérer dans la distribution
    packages=find_packages(),

    author="Nicolas CELLIER",
    author_email="contact@nicolas-cellier.net",
    description="Python library for falling-films modeling",

    # long_description=open('README.md').read(),

    # ext_modules=ext_modules,
    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,

    # Une url qui pointe vers la page officielle de votre lib
    # url='http://github.com/celliern/numerical-wave',

    # Il est d'usage de mettre quelques metadata à propos de sa lib
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers.

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 2 - Development",
        "License :: GNU GPL V3",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Topic :: Physical modeling",
    ],
)
