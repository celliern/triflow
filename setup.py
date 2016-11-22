#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='triflow',
    # la version du code
    version="0.2.1",
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

    entry_points={
        'console_scripts':
        ['datreant_server='\
         'triflow.plugins.writers.remote:datreant_server_writer',
         'triflow_cache_full='\
         'triflow.models.cache_main_models:cache_full',
         'triflow_cache_simple='\
         'triflow.models.cache_main_models:cache_simple'],
    },

    # Une url qui pointe vers la page officielle de votre lib
    # url='http://github.com/celliern/numerical-wave',

    # Il est d'usage de mettre quelques metadata à propos de sa lib
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers.

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)
