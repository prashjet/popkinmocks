.. _install:

Installation
============

.. note:: ``popkinmocks`` requires Python 3.7 and later.

This package is tested on Python versions 3.7 and 3.11 on Ubuntu and MacOS systems.

From Source
-----------

The source code for *popkinmocks* can be downloaded and installed `from GitHub <https://github.com/prashjet/popkinmocks>`_ by running

.. code-block:: bash

    git clone https://github.com/prashjet/popkinmocks.git
    cd popkinmocks
    python -m pip install -e .


Testing
-------

To run the unit tests, install the development dependencies using pip:

.. code-block:: bash

    python -m pip install -e ".[testing]"

and then within the main directory execute:

.. code-block:: bash

    pytest

All of the tests should pass (though warnings about "invalid value encountered in subtract" may appear).
If any of the tests don't pass and if you can't sort out why, `open an issue on GitHub <https://github.com/prashjet/popkinmocks/issues>`_.
