===========
Integrating
===========


The Harpia project has a ``CUDA/c++`` internal layer and a ``Python`` external layer.
The integration of new functionalities must attend to this structure.

CUDA/c++ layer
==============

1. Add files to scr/
--------------------

You may choose the best location in the src/ folder to place the ``CUDA/c++`` 
files. We will use the implementation of the anisotropicDiffusion2DGPU as an 
example.

Some importante points:

* You will probabbly need two files, with ``.cu`` and ``.h`` extensions. They must have the same name
* Harpia project only accepts ``.cu`` extension, whether it is a c, c++ or cuda code
* Your function must have a template for images, since we deal with different image representations
**Note:** remember to include the header (``.h``) in the ``.cu`` file

Templates
---------

A template is a c++ tool to avoid writting the same code 
for different data types. The data type is passed as a parameter to the function.
For example:

.. code-block:: c++

    template<typename dtype>
    void anisotropicDiffusion2D(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols) {
                            //function code
                            }
    template  void anisotropicDiffusion2DGPU<float>(float*,   int, float, float, int, int, int);
    template  void anisotropicDiffusion2DGPU<double>(double*, int, float, float, int, int, int);

In this example, dtype can represent eather a float or a double type.


Python layer
============

3. Cython wrapper
-----------------

The python wrapper must be defined in an appropriete file from sscPySpin. For 
the anisotropicDiffusion2DGPU, which represnets a filter, the suited file 
is: ``harpia/python/sscPySpin/filters/filters.pyx``. It will be needed a 
``cdef`` and a ``def`` declaration for the new function.

cdef
----

cdef is used for Cython functions that are intended to be pure ‘C’ functions. 
All types must be declared. Cython aggressively optimises the the code and there 
are a number of gotchas. The generated code is about as fast as you can get 
though.

cdef declared functions are not visible to Python code that imports the module. 

Take some care with cdef declared functions; it looks like you are writing 
Python but actually you are writing C. (`Reference 
<https://notes-on-cython.readthedocs.io/en/latest/function_declarations.html>`_)

.. code-block:: c++

    cdef extern from '../../../src/filters/anisotropic_diffusion.h':
        void anisotropicDiffusion2D[dtype](dtype* inputImage, int totalIterations, float deltaT, 
                                float kappa, int diffusionOption, int numRows, int numCols)

def
---
.. line length------------------------------------------------------------------

def is used for code that will be:

* Called directly from Python code with Python objects as arguments
* Returns a Python object

The generated code treats every operation as if it was dealing with Python 
objects with Python consequences so it incurs a high overhead. Declaring the 
types of arguments and local types (thus return values) can allow Cython to 
generate optimised code which speeds up the execution. If the types are declared 
then a TypeError will be raised if the function is passed the wrong types. 
(`Reference 
<https://notes-on-cython.readthedocs.io/en/latest/function_declarations.html>`_)

.. code-block:: python

    def  anisotropic2D(numeric[:,::1] inputImage, int totalIterations, float deltaT, 
                                float kappa, int diffusionOption):
        
        #these inputs are requested by the c++ function, but do not need to be python inputs
        cdef int rows = inputImage.shape[0]
        cdef int cols = inputImage.shape[1]

        anisotropicDiffusion2D(ref_2d[numeric](inputImage), totalIterations, deltaT, 
                                kappa, diffusionOption, rows, cols)



