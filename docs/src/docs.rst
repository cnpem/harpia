====
Docs
====

Install tools
-------------

First of all, it is necessary to install Doxygen, Sphinx, Read the Docs theme and Breathe. The commands are:

.. code-block:: bash
    
    sudo apt install doxygen
    apt-get install python3-sphinx
    pip install sphinx-rtd-theme
    pip install breathe

Build docs
----------

Once the docstrings are written, the documentation is generated using the following commands inside `docs/`:

.. code-block:: bash

    sphinx-apidoc -o src/harpia ../harpia
    doxygen Doxyfile.in
    make

The folder `build/html/`, containing the documentation, will be created and can be seen on any browser.

.. code-block:: bash

    firefox build/html/index.html

.. note:: 
    
    The folder `build/` is untracked by Git because it is too large.