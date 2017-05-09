Installation
============


External requirements
---------------------

The library is based on Theano, thus extra dependecies like fortran and C compiler are needed, see Theano install page for extra informations:

http://deeplearning.net/software/theano/install.html


via PyPI
---------

Beware, the PyPI version is not always up-to-date.

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

You can install the last version of the library using pip and the github repository:

.. code:: bash

    pip install git+git://github.com/locie/triflow.git
