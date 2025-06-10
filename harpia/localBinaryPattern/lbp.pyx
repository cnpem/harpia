cimport cython
cimport numpy
import numpy

from libcpp cimport bool
from harpia.common import Size

#Define the fused type for numeric types : float, int, unsigned int
ctypedef fused numeric:
    float
    int
    unsigned int
#---------------------------------------------------------------------------------------------------
cdef extern from "../../include/localBinaryPattern/lbp.h":
    void localBinaryPattern(numeric* hostImage, float* hostOutput, int xsize, int ysize, int zsize)
