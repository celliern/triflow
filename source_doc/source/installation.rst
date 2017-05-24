Installation
===============


External requirements
---------------------

This library is written for python >= 3.6, and I recommend to install it via Anaconda3_ : this is a full python distribution including a scientific-oriented IDE, the main scientific python libraries and the Jupyter project.

The library is based on Theano, thus extra dependecies like fortran and C compiler are needed, see Theano install page for extra informations:

http://deeplearning.net/software/theano/install.html


via PyPI
---------

.. code:: bash

    pip install triflow

will install the package and

.. code:: bash

    pip install triflow --upgrade

will update an old version of the library.

use sudo if needed, and the user flag if you want to install it without the root privileges:

.. code:: bash

    pip install --user triflow

via github
-----------

You can install the last version of the library using pip and the `github repository`_:

.. code:: bash

    pip install git+git://github.com/locie/triflow.git


.. _github repository: https://github.com/locie/triflow
.. _Anaconda3: https://www.continuum.io/downloads
