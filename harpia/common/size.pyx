import numpy as np

cimport numpy as np


cdef class Size:
    # Available in Python-space, but only for reading:
    cdef readonly int x, y, z  # Declare x, y, z as class attributes
    
    def __init__(self, input_array):
        size = input_array.shape

        self.y = size[0] #ysize
        self.x = size[1] #xsize
        self.z = size[2] #zsize

        '''xsize is the number of elements in the x direction. In np.ndarray it is equivalent to 
        the number of columns and is represented as the second dimension in the shape.'''
