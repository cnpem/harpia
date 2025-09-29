import numpy as np

cimport numpy as np

cdef class Size:
    # Available in Python-space, but only for reading:
    cdef readonly int x, y, z  # Declare x, y, z as class attributes
    
    def __init__(self, input_array):
        if input_array is None or input_array.size == 0:
            raise ValueError("Input array is None or empty. Expected a 3-dimensional array.")
            
        size = input_array.shape

        if(len(size)==3): 
            self.x = size[2] #xsize
            self.y = size[1] #ysize
            self.z = size[0] #zsize
        elif(len(size)==2):
            self.x = size[1] #xsize
            self.y = size[0] #ysize
            self.z = 1       #zsize
        else:
            raise ValueError(f"Incompatible size. Expected 3 or 2 dimensions, but received array with {len(size)} dimensions.")
