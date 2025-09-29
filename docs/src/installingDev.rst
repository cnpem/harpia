==========
Installing
==========
.. line length------------------------------------------------------------------

Harpia consists of a c++/cuda framework with python wrapper. The module 
installation can be done from the python layer

.. _harpia-dev-installation:

Using Docker Container
======================

1. Build the docker image

.. code-block:: bash

  docker build -f Dockerfile.dev -t harpia_dev . 

2. Run a docker container

.. code-block:: bash

 docker run --gpus all -p 8888:8888 -it -v "$(pwd)":/workspace/harpia harpia_dev

.. note:: 

  The dockerfile volume allows you to see the local files inside the container, 
  they are the same files.   Any changes made to the files in your local 
  filesystem will also be seen inside the container.

3. Inside the container, you can build harpia library

.. code-block:: bash
  
  cd harpia
  python3 setup.py

4. To test the built library, you can call python in the terminal and import 
the module

.. code-block:: bash
  
  python3 

.. code-block:: python

  import harpia

5. Once inside the container, start the Jupyter Notebook server:

.. code-block:: bash
  cd tests_python
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

This command will start the Jupyter Notebook server, listening on all interfaces 
(--ip=0.0.0.0) on port 8888 (--port=8888). The --no-browser option prevents 
Jupyter from trying to open a browser, and --allow-root allows it to run as root 
(necessary in many Docker containers).

6. Access Jupyter Notebook: Open your web browser and navigate to 
http://localhost:8888. You will see the Jupyter Notebook interface. You may need
to copy the token from the terminal output to log in.


Installing locally
==================

1. Install requirements

.. code-block:: bash

  pip install -r requirements.txt

2. Build project with cython wrapper

.. code-block:: bash

  python3 setup.py

3. Test in python level

.. code-block:: bash

  cd tests_python
  python3 test_script.py


Testing on C++ level
====================

1. Build library

.. code-block:: bash

  make

1. Execute tests

.. code-block:: bash

  .\Test