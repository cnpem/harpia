cimport cython
cimport numpy as np
import numpy as np

from libcpp cimport bool
from harpia.common import Size

#Define the fused type for numeric types : float, double
ctypedef fused real:
    float
    double

cdef extern from '../include/filters/anisotropic_diffusion.h':

    void anisotropicDiffusion2DGPU[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize)                        

    void anisotropicDiffusion3D[dtype](dtype* hostImage, dtype* hostOutput, int totalIterations, float deltaT, float kappa,
                               int diffusionOption, int xsize, int ysize, int zsize, const int flag_verbose, float gpuMemory, int ngpus)


def anisotropic_diffusion2D(np.ndarray[real, ndim=2] image, int num_iter,
                          float delta_t, float kappa, int diffusion_option):
    """
    Performs anisotropic diffusion on a 2D image.

    This function applies the anisotropic diffusion algorithm to enhance images by reducing noise while preserving edges.
    It supports three different diffusion options that control the smoothing behavior.

    Parameters:
    -----------
    image : float numpy.ndarray
        The input 2D image data.
    num_iter : int
        Number of iterations to perform.
    delta_t : float
        Time step size.
    kappa : float
        Gradient modulus threshold that influences the conduction.
    diffusion_option : int
        Choice of diffusion function:
        - 1: Exponential decay
        - 2: Inverse quadratic decay
        - 3: Hyperbolic tangent decay
          Option 3 is a faster implementation based on:
          Mbarki, Zouhair, et al. "A new rapid auto-adapting diffusion function for adaptive anisotropic 
          image de-noising and sharply conserved edges." Computers & Mathematics with Applications 74.8 (2017): 1751-1768.

    Returns:
    --------
    output_image
        The diffused image in the same data type as the input..
    """

    # Define variables
    isize = Size(image)
    
    # Create the output array
    cdef np.ndarray[real, ndim=2] hostOutput = np.empty_like(image)

    anisotropicDiffusion2DGPU(&image[0,0], &hostOutput[0,0], num_iter, delta_t, kappa, diffusion_option, isize.x, isize.y)

    return hostOutput

def anisotropic_diffusion3D(np.ndarray[real, ndim=3] image, int num_iter,
                          float delta_t, float kappa, int diffusion_option, int flag_verbose, float gpuMemory, int ngpus = -1):
    """
    Performs anisotropic diffusion on a 3D image.

    This function applies the anisotropic diffusion algorithm to enhance images by reducing noise while preserving edges.
    It supports three different diffusion options that control the smoothing behavior.

    Parameters:
    -----------
    image : float numpy.ndarray
        The input 3D image data.
    num_iter : int
        Number of iterations to perform.
    delta_t : float
        Time step size.
    kappa : float
        Gradient modulus threshold that influences the conduction.
    diffusion_option : int
        Choice of diffusion function:
        - 1: Exponential decay
        - 2: Inverse quadratic decay
        - 3: Hyperbolic tangent decay
          Option 3 is a faster implementation based on:
          Mbarki, Zouhair, et al. "A new rapid auto-adapting diffusion function for adaptive anisotropic 
          image de-noising and sharply conserved edges." Computers & Mathematics with Applications 74.8 (2017): 1751-1768.
    flag_verbose: int
        Verbose for number of chuncks in execution
    gpuMemmory: bool
        Percentage of memmory occupied in the GPU (if using the gpu function). With cython value, working value is of 0.4 (40%).
    ngpus: int 
        The number of GPUs to use. 
        If ngpus < 1, all available GPUs are used.
        If ngpus = 0, CPU execution is selected. 
        If ngpus >= 1, the function uses up to min(ngpus, available GPUs).

    Returns:
    --------
    output_image
        The diffused image in the same data type as the input.
    """
    # Define variables
    isize = Size(image)
    
    # Create the output array
    cdef np.ndarray[real, ndim=3] hostOutput = np.empty_like(image)

    anisotropicDiffusion3D(&image[0,0,0], &hostOutput[0,0,0], num_iter, delta_t, kappa, 
    diffusion_option, isize.x, isize.y, isize.z, flag_verbose, gpuMemory, ngpus)

    return hostOutput

