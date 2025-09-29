==============
Repo Structure
==============

The harpia project is better underestood by deviding its complexity layers. The
outer layer is written in pyhton with a cython wrapper. This allows to import
compiled c++/cuda code in pyhton, merging the advantages of efficient c++/cuda 
implementation and python accecibility. The inner layer is written in c++/cuda. 
The major algorithms for fast and efficient image analysis are written in this layer. 

The project is intended to be distribuited in Python, but it can be built, used and tested 
at c++ level also. 

Python layer
------------

In `harpia/` folder is placed the cython wrapper code, that exposes all compiled 
functionalities to be imported at python level. Test scripts for python usage
can be found in the folder `tests_python/`. The project can be built using setup.py.

.. code-block:: none

    docs/

    harpia/
        filters/
        morphology/
        quantification/
        threshold/
    
    tests_python/
        morphology/

    setup.py


c++/cuda layer
--------------

The main functionalities of harpia are written in CUDA for GPU optimization. 
The source code can be found in the folders `src/` and `include/`. Tests at c++ level
are placed in the folder `tests_cuda/`, which can be build from the Makefile.

.. code-block:: none

    include/
        common/
        filters/
        morphology/
        quantification/
        threshold/

    src/
        filters/
        morphology/
        quantification/
        threshold/
    
    tests_cuda/
        morphology/
        main.cpp

    Makefile



