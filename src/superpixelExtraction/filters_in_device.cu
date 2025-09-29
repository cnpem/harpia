#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

#include "../../include/filters/gaussian_filter.h"
#include "../../include/filters/prewitt_filter.h"
#include "../../include/localBinaryPattern/lbp.h"
#include "../../include/morphology/cuda_helper.h"
#include "../../include/superpixelExtraction/filters_in_device.h"

// Helper function to apply Gaussian filter
void applyGaussianFilterDevice2D(
    float* d_image,
    float* d_image_smoothed,
    float sigma,
    int xsize, int ysize, int zsize) {
    
    // Generate Gaussian kernel
    int ksize = static_cast<int>(std::ceil(4.0f * sigma + 0.5f));
    if (ksize % 2 == 0) ksize++; // Ensure odd kernel size
    
    double* h_kernel = nullptr;
    get_gaussian_kernel_2d(&h_kernel, ksize, ksize, sigma);
    
    double* d_kernel;
    CHECK(cudaMalloc(&d_kernel, ksize * ksize * sizeof(double)));
    CHECK(cudaMemcpy(d_kernel, h_kernel, ksize * ksize * sizeof(double), cudaMemcpyHostToDevice));
    free(h_kernel);
    
    // Configure 2D grid and block sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((ysize + blockSize.x - 1) / blockSize.x, (xsize + blockSize.y - 1) / blockSize.y);
    
    // Apply filter slice by slice
    for (int k = 0; k < zsize; ++k) {
        gaussian_filter_kernel_2d<<<gridSize, blockSize>>>(
            d_image, d_image_smoothed, d_kernel, k,
            ysize, xsize, zsize, ksize, ksize
        );
    }
    CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_kernel);
}

// Helper function to apply Prewitt filter
void applyPrewittFilterDevice2D(
    float* d_image_smoothed,
    float* d_temp_image,
    int xsize, int ysize, int zsize) {
    
    float* kernelHorizontal;
    get_prewitt_horizontal_kernel_2d(&kernelHorizontal);

    float* kernelVertical;
    get_prewitt_vertical_kernel_2d(&kernelVertical);

    float* deviceKernelHorizontal;
    CHECK(cudaMalloc((void**)&deviceKernelHorizontal, 9 * sizeof(float)));
    CHECK(cudaMemcpy(deviceKernelHorizontal, kernelHorizontal, 9 * sizeof(float), cudaMemcpyHostToDevice));

    float* deviceKernelVertical;
    CHECK(cudaMalloc((void**)&deviceKernelVertical, 9 * sizeof(float)));
    CHECK(cudaMemcpy(deviceKernelVertical, kernelVertical, 9 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32);
    dim3 gridSize((ysize + blockSize.x - 1) / blockSize.x, (xsize + blockSize.y - 1) / blockSize.y);

    // Apply Prewitt filter slice by slice
    for (int k = 0; k < zsize; ++k) {
        prewitt_filter_kernel_2d<<<gridSize, blockSize>>>(
            d_image_smoothed, d_temp_image,
            deviceKernelHorizontal, deviceKernelVertical, k, ysize, xsize,
          zsize);
    }

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(deviceKernelHorizontal));
    CHECK(cudaFree(deviceKernelVertical));
    free(kernelHorizontal);
    free(kernelVertical);
}

// Helper function to apply Local Binary Pattern
void applyLocalBinaryPatternDevice2D(
    float* d_image_smoothed,
    float* d_temp_image,
    int xsize, int ysize, int zsize) {
    
    // Configure 2D grid and block sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((ysize + blockSize.x - 1) / blockSize.x, (xsize + blockSize.y - 1) / blockSize.y);
    
    // Apply LBP slice by slice
    for (int k = 0; k < zsize; ++k) {
        lbp<<<gridSize, blockSize>>>(
            d_image_smoothed, d_temp_image, ysize, xsize, zsize, k
        );
    }
    CHECK(cudaDeviceSynchronize());
}
//=============  Hessian and Shape Index Computation =============
// Device function to compute second derivative at a specific point
__device__ float compute_second_derivative_at_point(float* d_image, int x, int y, int z,
                                                   int xsize, int ysize, int zsize,
                                                   int axis, int step) {
    int idx = z * xsize * ysize + y * xsize + x;
    
    if (axis == 0) { // second derivative along x (d²f/dx²)
        if (x + step < xsize && x - step >= 0) {
            int idx_plus = z * xsize * ysize + y * xsize + (x + step);
            int idx_minus = z * xsize * ysize + y * xsize + (x - step);
            return (d_image[idx_plus] - 2.0f * d_image[idx] + d_image[idx_minus]) / (step * step);
        }
        // From here is just border case handling 
        else if (x - step < 0 && x + 3 * step < xsize) {
            // Forward difference at left boundary
            int idx_0 = z * xsize * ysize + y * xsize + x;
            int idx_1 = z * xsize * ysize + y * xsize + (x + step);
            int idx_2 = z * xsize * ysize + y * xsize + (x + 2 * step);
            int idx_3 = z * xsize * ysize + y * xsize + (x + 3 * step);
            return (2.0f * d_image[idx_0] - 5.0f * d_image[idx_1] + 4.0f * d_image[idx_2] - d_image[idx_3]) / (step * step * step);
        } else if (x + step >= xsize && x - 3 * step >= 0) {
            // Backward difference at right boundary
            int idx_0 = z * xsize * ysize + y * xsize + x;
            int idx_1 = z * xsize * ysize + y * xsize + (x - step);
            int idx_2 = z * xsize * ysize + y * xsize + (x - 2 * step);
            int idx_3 = z * xsize * ysize + y * xsize + (x - 3 * step);
            return (2.0f * d_image[idx_0] - 5.0f * d_image[idx_1] + 4.0f * d_image[idx_2] - d_image[idx_3]) / (step * step * step);
        }
    } else if (axis == 1) { // second derivative along y (d²f/dy²)
        if (y + step < ysize && y - step >= 0) {
            int idx_plus = z * xsize * ysize + (y + step) * xsize + x;
            int idx_minus = z * xsize * ysize + (y - step) * xsize + x;
            return (d_image[idx_plus] - 2.0f * d_image[idx] + d_image[idx_minus]) / (step * step);
        }
        else if (y - step < 0 && y + 3 * step < ysize) {
            // Forward difference at top boundary
            int idx_0 = z * xsize * ysize + y * xsize + x;
            int idx_1 = z * xsize * ysize + (y + step) * xsize + x;
            int idx_2 = z * xsize * ysize + (y + 2 * step) * xsize + x;
            int idx_3 = z * xsize * ysize + (y + 3 * step) * xsize + x;
            return (2.0f * d_image[idx_0] - 5.0f * d_image[idx_1] + 4.0f * d_image[idx_2] - d_image[idx_3]) / (step * step * step);
        } else if (y + step >= ysize && y - 3 * step >= 0) {
            // Backward difference at bottom boundary
            int idx_0 = z * xsize * ysize + y * xsize + x;
            int idx_1 = z * xsize * ysize + (y - step) * xsize + x;
            int idx_2 = z * xsize * ysize + (y - 2 * step) * xsize + x;
            int idx_3 = z * xsize * ysize + (y - 3 * step) * xsize + x;
            return (2.0f * d_image[idx_0] - 5.0f * d_image[idx_1] + 4.0f * d_image[idx_2] - d_image[idx_3]) / (step * step * step);
        }
    }
    
    return 0.0f;
}

__device__ float compute_first_derivative_backward_forward(float* d_image, int x, int y, int z,
                                          int xsize, int ysize, int zsize,
                                          int axis, int step,
                                          int direction) {
    int offset = z * xsize * ysize;

    // Backward difference
    if (direction == -1) {
        if (axis == 0) { // x-axis
            if (x - 2 * step >= 0) {
                int idx0 = offset + y * xsize + x;
                int idx1 = offset + y * xsize + (x - step);
                int idx2 = offset + y * xsize + (x - 2 * step);
                return (3.0f * d_image[idx0] - 4.0f * d_image[idx1] + d_image[idx2]) / (2.0f * step);
            }
        } else if (axis == 1) { // y-axis
            if (y - 2 * step >= 0) {
                int idx0 = offset + y * xsize + x;
                int idx1 = offset + (y - step) * xsize + x;
                int idx2 = offset + (y - 2 * step) * xsize + x;
                return (3.0f * d_image[idx0] - 4.0f * d_image[idx1] + d_image[idx2]) / (2.0f * step);
            }
        }
    }
    // Forward difference (direction == +1)
    else {
        if (axis == 0) { // x-axis
            if (x + 2 * step < xsize) {
                int idx0 = offset + y * xsize + x;
                int idx1 = offset + y * xsize + (x + step);
                int idx2 = offset + y * xsize + (x + 2 * step);
                return (-3.0f * d_image[idx0] + 4.0f * d_image[idx1] - d_image[idx2]) / (2.0f * step);
            }
        } else if (axis == 1) { // y-axis
            if (y + 2 * step < ysize) {
                int idx0 = offset + y * xsize + x;
                int idx1 = offset + (y + step) * xsize + x;
                int idx2 = offset + (y + 2 * step) * xsize + x;
                return (-3.0f * d_image[idx0] + 4.0f * d_image[idx1] - d_image[idx2]) / (2.0f * step);
            }
        }
    }

    // Out of bounds
    return 0.0f;
}

// Device function to compute mixed partial derivative (d²f/dxdy)
__device__ float compute_mixed_derivative_at_point(float* d_image, int x, int y, int z,
                                                   int xsize, int ysize, int zsize, int step) {
    int offset = z * xsize * ysize;

    // Centered difference (interior) - most accurate
    if (x + step < xsize && x - step >= 0 && y + step < ysize && y - step >= 0) {
        int idx_pp = offset + (y + step) * xsize + (x + step); // f(y+1,x+1)
        int idx_pm = offset + (y + step) * xsize + (x - step); // f(y+1,x-1)
        int idx_mp = offset + (y - step) * xsize + (x + step); // f(y-1,x+1)
        int idx_mm = offset + (y - step) * xsize + (x - step); // f(y-1,x-1)

        return (d_image[idx_pp] - d_image[idx_pm] - d_image[idx_mp] + d_image[idx_mm]) /
               (4.0f * step * step);
    } 
    // x boundary (x backward/forward and y centered)
    else if ((x + step >= xsize || x - step < 0) && y + step < ysize && y - step >= 0) {
        // forward by default, change to backward if at right boundary
        int direction = 1;
        if (x + step >= xsize) {
            direction = -1;
        }
        
        // Check if we have enough points for the boundary method
        if ((direction == 1 && x + 2 * step >= xsize) || 
            (direction == -1 && x - 2 * step < 0)) {
            return 0.0f; // Not enough points
        }
        
        // calculate forward/backward for each centered term of the equation
        float fxyp = compute_first_derivative_backward_forward(d_image, x, y + step, z, 
                                                              xsize, ysize, zsize, 0, step, direction);
        float fxyn = compute_first_derivative_backward_forward(d_image, x, y - step, z, 
                                                              xsize, ysize, zsize, 0, step, direction);
        return (fxyp - fxyn) / (2.0f * step);
    }
    // y boundary (x centered and y backward/forward)
    else if (x + step < xsize && x - step >= 0 && (y + step >= ysize || y - step < 0)) {
        // forward by default
        int direction = 1;
        // change to backward if we are at top boundary
        if (y + step >= ysize) {
            direction = -1;
        }
        
        // Check if we have enough points for the boundary method
        if ((direction == 1 && y + 2 * step >= ysize) || 
            (direction == -1 && y - 2 * step < 0)) {
            return 0.0f; // Not enough points
        }
        
        float fyxp = compute_first_derivative_backward_forward(d_image, x + step, y, z, 
                                                              xsize, ysize, zsize, 1, step, direction);
        float fyxn = compute_first_derivative_backward_forward(d_image, x - step, y, z, 
                                                              xsize, ysize, zsize, 1, step, direction);

        return (fyxp - fyxn) / (2.0f * step);
    } 
    // corner i.e xy boundary (x forward/backward and y forward/backward)
    else if ((x + step >= xsize || x - step < 0) && (y + step >= ysize || y - step < 0)) {
        // forward by default, change if at boundary
        int directiony = 1;
        if (y + step >= ysize) {
            directiony = -1;
        }

        int directionx = 1;
        if (x + step >= xsize) {
            directionx = -1;
        }

        // Check if we have enough points for the boundary method
        if ((directionx == 1 && x + 2 * step >= xsize) || 
            (directionx == -1 && x - 2 * step < 0) ||
            (directiony == 1 && y + 2 * step >= ysize) || 
            (directiony == -1 && y - 2 * step < 0)) {
            return 0.0f; // Not enough points
        }
        
        // calculate forward/backward in x
        // df(x,y)/dx
        float fxy = compute_first_derivative_backward_forward(d_image, x, y, z, 
                                                             xsize, ysize, zsize, 0, step, directionx);
        // df(x,y+∆y)/dx
        float fxyp = compute_first_derivative_backward_forward(d_image, x, y + directiony * step, z, 
                                                              xsize, ysize, zsize, 0, step, directionx);
        // df(x,y+2∆y)/dx
        float fxyp2 = compute_first_derivative_backward_forward(d_image, x, y + 2 * directiony * step, z, 
                                                               xsize, ysize, zsize, 0, step, directionx);

        // Apply finite difference in y direction
        if (directiony == 1) {
            // Forward difference: (-3f₀ + 4f₁ - f₂) / (2Δy)
            return (-3.0f * fxy + 4.0f * fxyp - fxyp2) / (2.0f * step);
        } else {
            // Backward difference: (3f₀ - 4f₋₁ + f₋₂) / (2Δy)
            return (3.0f * fxy - 4.0f * fxyp + fxyp2) / (2.0f * step);
        }
    }
    
    return 0.0f;
}

// CUDA kernel for computing Hessian eigenvalues directly from image
__global__ void hessian_eigenvalues_kernel(float* d_image, float* d_eigen1, float* d_eigen2,
                                         int xsize, int ysize, int zsize, 
                                         int slice_idx, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= xsize || y >= ysize) return;
    
    int idx = slice_idx * xsize * ysize + y * xsize + x;
    
    // Compute Hessian components on-the-fly
    float dfdxdx = compute_second_derivative_at_point(d_image, x, y, slice_idx, 
                                                     xsize, ysize, zsize, 0, step);
    float dfdydy  = compute_second_derivative_at_point(d_image, x, y, slice_idx, 
                                                     xsize, ysize, zsize, 1, step);
    float dfdxdy  = compute_mixed_derivative_at_point(d_image, x, y, slice_idx, 
                                                    xsize, ysize, zsize, step);
    // Compute eigenvalues
    float trace = dfdxdx + dfdydy;
    float delta = (dfdxdx - dfdydy) * (dfdxdx - dfdydy) + 4.0f * dfdxdy * dfdxdy;
    float sqrt_delta = sqrtf(delta);
    
    float lambda1 = (trace + sqrt_delta) * 0.5f;
    float lambda2 = (trace - sqrt_delta) * 0.5f;
    
    // Store eigenvalues in interleaved format
    d_eigen1[idx] = lambda1;
    d_eigen2[idx] = lambda2;
}

// CUDA kernel for computing shape index directly from image
__global__ void shape_index_kernel(float* d_image, float* d_shape_index,
                                 int xsize, int ysize, int zsize, 
                                 int slice_idx, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= xsize || y >= ysize) return;
    
    int idx = slice_idx * xsize * ysize + y * xsize + x;
    
    // Compute Hessian components on-the-fly
    float dfdxdx = compute_second_derivative_at_point(d_image, x, y, slice_idx, 
                                                     xsize, ysize, zsize, 0, step);
    float dfdydy = compute_second_derivative_at_point(d_image, x, y, slice_idx, 
                                                     xsize, ysize, zsize, 1, step);
    float dfdxdy = compute_mixed_derivative_at_point(d_image, x, y, slice_idx, 
                                                    xsize, ysize, zsize, step);
    
    // Compute eigenvalues
    float trace = dfdxdx + dfdydy;
    float delta = (dfdxdx - dfdydy) * (dfdxdx - dfdydy) + 4.0f * dfdxdy * dfdxdy;
    float sqrt_delta = sqrtf(delta);
    
    float l1 = (trace + sqrt_delta) * 0.5f;
    float l2 = (trace - sqrt_delta) * 0.5f;
    
    // Ensure l1 >= l2
    if (l1 < l2) {
        float temp = l1;
        l1 = l2;
        l2 = temp;
    }
    
    // Compute shape index
    float num = l2 + l1;
    float den = l2 - l1;
    
    float sidx;
    if (fabsf(den) < 1e-8f) {
        sidx = 0.0f;
    } else {
        sidx = (2.0f / M_PI) * atanf(num / den);
    }
    
    d_shape_index[idx] = sidx;
}

// Host function to compute Hessian eigenvalues
void applyHessianEigenvaluesDevice2D(float* d_image, float* d_eigen1, float* d_eigen2,
                        int xsize, int ysize, int zsize, int step) {
    
    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, 
                  (ysize + blockSize.y - 1) / blockSize.y);
    
    // Compute eigenvalues for each slice
    for (int idz = 0; idz < zsize; ++idz) {
        hessian_eigenvalues_kernel<<<gridSize, blockSize>>>(
            d_image, d_eigen1, d_eigen2, xsize, ysize, zsize, idz, step);
        
        cudaDeviceSynchronize();
    }
}

// Host function to compute shape index
void applyShapeIndexDevice2D(float* d_image, float* d_shape_index, 
                int xsize, int ysize, int zsize, int step) {
    
    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, 
                  (ysize + blockSize.y - 1) / blockSize.y);
    
    // Compute shape index for each slice
    for (int idz = 0; idz < zsize; ++idz) {
        shape_index_kernel<<<gridSize, blockSize>>>(
            d_image, d_shape_index, xsize, ysize, zsize, idz, step);
        
        cudaDeviceSynchronize();
    }
}

// Combined kernel that computes both eigenvalues and shape index in one pass
__global__ void hessian_analysis_kernel(float* d_image, float* d_eigen1, float* d_eigen2, float* d_shape_index,
                                       int xsize, int ysize, int zsize, 
                                       int slice_idx, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= xsize || y >= ysize) return;
    
    int idx = slice_idx * xsize * ysize + y * xsize + x;
    
    // Compute Hessian components on-the-fly
    float dfdxdx = compute_second_derivative_at_point(d_image, x, y, slice_idx, 
                                                     xsize, ysize, zsize, 0, step);
    float dfdydy = compute_second_derivative_at_point(d_image, x, y, slice_idx, 
                                                     xsize, ysize, zsize, 1, step);
    float dfdxdy = compute_mixed_derivative_at_point(d_image, x, y, slice_idx, 
                                                    xsize, ysize, zsize, step);
    
    // Compute eigenvalues
    float trace = dfdxdx + dfdydy;
    float delta = (dfdxdx - dfdydy) * (dfdxdx - dfdydy) + 4.0f * dfdxdy * dfdxdy;
    float sqrt_delta = sqrtf(delta);
    
    float l1 = (trace + sqrt_delta) * 0.5f;
    float l2 = (trace - sqrt_delta) * 0.5f;

    // Store eigenvalues
    d_eigen1[idx] = l1;
    d_eigen2[idx] = l2;
    
    // Ensure l1 >= l2 for shape index computation
    if (l1 < l2) {
        float temp = l1;
        l1 = l2;
        l2 = temp;
    }
    
    // Compute shape index
    float num = l2 + l1;
    float den = l2 - l1;
    
    float sidx;
    if (fabsf(den) < 1e-8f) {
        sidx = 0.0f;
    } else {
        sidx = (2.0f / M_PI) * atanf(num / den);
    }
    
    d_shape_index[idx] = sidx;
}

// Host function to compute both eigenvalues and shape index in one pass
void hessian_analysis(float* d_image, float* d_eigen1, float* d_eigen2, float* d_shape_index,
                     int xsize, int ysize, int zsize, int step) {
    
    dim3 blockSize(32, 32);
    dim3 gridSize((xsize + blockSize.x - 1) / blockSize.x, 
                  (ysize + blockSize.y - 1) / blockSize.y);
    
    // Compute both eigenvalues and shape index for each slice
    for (int idz = 0; idz < zsize; ++idz) {
        hessian_analysis_kernel<<<gridSize, blockSize>>>(
            d_image, d_eigen1, d_eigen2, d_shape_index, xsize, ysize, zsize, idz, step);
        
        cudaDeviceSynchronize();
    }
}