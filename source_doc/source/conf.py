#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# triflow documentation build configuration file, created by
# sphinx-quickstart on Thu Jan 26 14:29:11 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

from recommonmark.parser import CommonMarkParser

extensions = ['matplotlib.sphinxext.mathmpl',
              'matplotlib.sphinxext.only_directives',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              #   'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.githubpages',
              'sphinxcontrib.napoleon',
              'IPython.sphinxext.ipython_console_highlighting',
              'nbsphinx']

napoleon_include_special_with_doc = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md', '.ipynb']

# The master toctree document.
master_doc = 'doc'

project = 'triflow'
copyright = '2017, Nicolas Cellier'
author = 'Nicolas Cellier'

version = '0.5.2'
language = "python"

exclude_patterns = ['_build', '**.ipynb_checkpoints']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_logo = '_static/images/logo_triflow_reduced.png'
html_favicon = '_static/images/favicon.ico'
html_extra_path = ['index.html']
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'triflowdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
}

latex_documents = [
    (master_doc, 'triflow.tex', 'triflow Documentation',
     'Nicolas Cellier', 'manual'),
]


# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, 'triflow', 'triflow Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (master_doc, 'triflow', 'triflow Documentation',
     author, 'triflow', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output ----------------------------------------------

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

source_parsers = {
    '.md': CommonMarkParser,
}

plot_pre_code = """
import numpy as np
import pylab as pl
pl.style.use("./publication.mplstyle")
"""
