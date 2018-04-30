Installation
===============


External requirements
---------------------

This library is written for python >= 3.5, and the library is based on Theano,
thus extra dependencies like fortran and C compiler are needed, see Theano
install page for extra informations:

http://deeplearning.net/software/theano/install.html

Because of these compiled-based dependencies, it can be useful to use
`Anaconda3`_ to install Theano.

via PyPI
---------

.. code:: bash

    pip install triflow

will install the package and

.. code:: bash

    pip install triflow --upgrade

will update an old version of the library.

use sudo if needed, and the user flag if you want to install it without the
root privileges:

.. code:: bash

    pip install --user triflow

via github
-----------

You can install the last version of the library using pip and the
`github repository`_:

.. code:: bash

    pip install git+git://github.com/locie/triflow.git

and for the developpement branch:

.. code:: bash

    pip install git+git://github.com/locie/triflow@dev

.. _github repository: https://github.com/locie/triflow
.. _Anaconda3: https://www.continuum.io/downloads
