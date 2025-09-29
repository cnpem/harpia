=====
Tests
=====
.. line length------------------------------------------------------------------

Here lies some useful tests for harpia package. Before testing, you will need to 
install the package.

.. _harpia-dev-installation:
.. _harpia-installation:

Tests on python level
=====================

1. After installing the harpia module, you may test in python level with the
commands:

.. code-block:: bash

  cd tests_python
  pyrhon3 test_script.py

Tests on c++/CUDA level
=======================

.. line length------------------------------------------------------------------

1. First, you need to build library with the approprieate compilers. 
This can be done using the Makefile. From the root of the repository, you can
run the following commands:

.. code-block:: bash

  make

1. The `Test` executable file will be created, to execute it you may run the 
command:

.. code-block:: bash

  .\Test