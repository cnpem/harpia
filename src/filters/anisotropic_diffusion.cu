#include <cmath>
#include <numeric>
#include"../../include/filters/anisotropic_diffusion.h"
#include <iostream>

template<typename dtype>
void anisotropicDiffusion2D(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols) {

    dtype nabla[8]; // Stores pixel values from the 8 cardinal directions
    dtype diffusionCoefficients[8]; // Weighted diffusion coefficients for each direction

    // Temporary buffer to hold updates during each iteration
    dtype* tempBuffer = new dtype[numRows * numCols];

    // Loop through each time iteration
    for (int iter = 0; iter < totalIterations; iter++) {
        // Iterate over each pixel in the image
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                int rowNorth = std::min(row + 1, numRows - 1);
                int rowSouth = std::max(row - 1, 0);
                int colWest  = std::max(col - 1, 0);
                int colEast  = std::min(col + 1, numCols - 1);

                // Calculate derivative for all directions
                dtype center = inputImage[row*numCols + col];

                nabla[0] = inputImage[rowNorth * numCols + col]     - center;  // North
                nabla[1] = inputImage[rowSouth * numCols + col]     - center;  // South
                nabla[2] = inputImage[row * numCols + colWest]      - center;   // West
                nabla[3] = inputImage[row * numCols + colEast]      - center;   // East
                nabla[4] = inputImage[rowNorth * numCols + colWest] - center; // Northwest
                nabla[5] = inputImage[rowNorth * numCols + colEast] - center; // Northeast
                nabla[6] = inputImage[rowSouth * numCols + colWest] - center; // Southwest
                nabla[7] = inputImage[rowSouth * numCols + colEast] - center; // Southeast

                // Calculate diffusion coefficients based on the option
                for (int i = 0; i < 8; i++) {
                    double scaledDiff = pow(nabla[i] / kappa, 2);
                    if (diffusionOption == 1) {
                        diffusionCoefficients[i] = nabla[i] * exp(-scaledDiff);
                    } else if (diffusionOption == 2) {
                        diffusionCoefficients[i] = nabla[i] / (1 + scaledDiff);
                    } else {
                        diffusionCoefficients[i] = nabla[i] * (1 - tanh(scaledDiff));
                    }
                }

                // Compute the sum of diffusion coefficients
                double diffusionSum = std::accumulate(diffusionCoefficients, diffusionCoefficients + 8, 0.0);
                // Update the output image
                tempBuffer[row * numCols + col] = inputImage[row * numCols + col] + deltaT * diffusionSum;
            }
        }
        // Update the values of the image for each iteration
        std::copy(tempBuffer, tempBuffer + numRows * numCols, inputImage);
    }
    delete [] tempBuffer;
}

template<typename dtype>
void anisotropicDiffusion3D(dtype* inputImage, int totalIterations, float deltaT,
                            float kappa, int diffusionOption, int numRows, int numCols, int numSlices) {

    dtype nabla[6]; // Stores pixel values from the 26 possible directions in 3D
    dtype diffusionCoefficients[6]; // Weighted diffusion coefficients for each direction

    // Temporary buffer to hold updates during each iteration
    dtype* tempBuffer = new dtype[numRows * numCols * numSlices];

    // Loop through each time iteration
    for (int iter = 0; iter < totalIterations; iter++) {
        // Iterate over each pixel in the image
        for (int slice = 0; slice < numSlices; slice++) {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    int idx = slice * numRows * numCols + row * numCols + col;
                    dtype center = inputImage[idx];

                    // Compute indices for the neighboring cells with boundary checks
                    int sliceFront = std::max(slice - 1, 0);
                    int sliceBack = std::min(slice + 1, numSlices - 1);
                    int rowNorth = std::min(row + 1, numRows - 1);
                    int rowSouth = std::max(row - 1, 0);
                    int colEast = std::min(col + 1, numCols - 1);
                    int colWest = std::max(col - 1, 0);

                    // Calculate derivatives for all directions in 3D
                    nabla[0] = inputImage[slice * numRows * numCols + rowNorth * numCols + col] - center; // North
                    nabla[1] = inputImage[slice * numRows * numCols + rowSouth * numCols + col] - center; // South
                    nabla[2] = inputImage[slice * numRows * numCols + row * numCols + colWest] - center; // West
                    nabla[3] = inputImage[slice * numRows * numCols + row * numCols + colEast] - center; // East
                    // Add the z-dimension derivatives
                    nabla[4] = inputImage[sliceFront * numRows * numCols + row * numCols + col] - center; // Front
                    nabla[5] = inputImage[sliceBack * numRows * numCols + row * numCols + col] - center; // Back

                    // There are no diagnal neighbors, for simplicity.

                    // Calculate diffusion coefficients based on the option
                    for (int i = 0; i < 6; i++) {
                        double scaledDiff = pow(nabla[i] / kappa, 2);
                        if (diffusionOption == 1) {
                            diffusionCoefficients[i] = nabla[i] * exp(-scaledDiff);
                        } else if (diffusionOption == 2) {
                            diffusionCoefficients[i] = nabla[i] / (1 + scaledDiff);
                        } else {
                            diffusionCoefficients[i] = nabla[i] * (1 - tanh(scaledDiff));
                        }
                    }

                    // Compute the sum of diffusion coefficients
                    double diffusionSum = std::accumulate(diffusionCoefficients, diffusionCoefficients + 6, 0.0);
                    // Update the output image
                    tempBuffer[idx] = center + deltaT * diffusionSum;
                }
            }
        }
        // Update the values of the image for each iteration
        std::copy(tempBuffer, tempBuffer + numRows * numCols * numSlices, inputImage);
    }
    delete[] tempBuffer;
}


// Explicit instantiation of the template for expected data types
template void anisotropicDiffusion2D<float>(float*, int, float, float, int, int, int);
template void anisotropicDiffusion2D<double>(double*, int, float, float, int, int, int);

// Explicit instantiation of the template for expected data types
template void anisotropicDiffusion3D<float>(float*, int, float, float, int, int, int, int);
template void anisotropicDiffusion3D<double>(double*, int, float, float, int, int, int, int);

template<typename dtype>
__global__ void anisotropicDiffusion2DKernel(dtype* inputImage, dtype* outputImage, float deltaT, 
                                           float kappa, int diffusionOption, int numRows, int numCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    if (row < numRows && col < numCols) {

        // Compute indices for the neighboring cells with boundary checks
        int rowNorth = min(row + 1, numRows - 1);
        int rowSouth = max(row - 1, 0);
        int colEast =  min(col + 1, numCols - 1);
        int colWest =  max(col - 1, 0);

        dtype center = inputImage[row * numCols + col];
        dtype nabla[8];
        double_t diffusionCoefficients[8];

        nabla[0] = inputImage[rowNorth * numCols + col] - center;  // North
        nabla[1] = inputImage[rowSouth * numCols + col] - center;  // South
        nabla[2] = inputImage[row * numCols + colWest]  - center;  // West
        nabla[3] = inputImage[row * numCols + colEast]  - center;  // East
        nabla[4] = inputImage[rowNorth * numCols + colWest] - center; // Northwest
        nabla[5] = inputImage[rowNorth * numCols + colEast] - center; // Northeast
        nabla[6] = inputImage[rowSouth * numCols + colWest] - center; // Southwest
        nabla[7] = inputImage[rowSouth * numCols + colEast] - center; // Southeast
        
        double_t diffusionSum = 0;
        
        for (int i = 0; i < 8; i++) {
            double scaledDiff = pow(nabla[i] / kappa, 2);
            if (diffusionOption == 1) {
                diffusionCoefficients[i] = nabla[i] * exp(-scaledDiff);
            } else if (diffusionOption == 2) {
                diffusionCoefficients[i] = nabla[i] / (1 + scaledDiff);
            } else {
                diffusionCoefficients[i] = nabla[i] * (1 - tanh(scaledDiff));
            }
            
            diffusionSum += diffusionCoefficients[i];
        
        }

        outputImage[row * numCols + col] = inputImage[row * numCols + col] + deltaT * diffusionSum;
    }
}


template<typename dtype>
void anisotropicDiffusion2DGPU(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols) {
    dtype *d_inputImage, *d_tempBuffer;
    size_t numBytes = numRows * numCols * sizeof(dtype);

    // Allocate memory for the input image on the device
    cudaMalloc((void**)&d_inputImage, numBytes);
    cudaMalloc((void**)&d_tempBuffer, numBytes);
    
    cudaMemcpy(d_inputImage, inputImage, numBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);

    for (int iter = 0; iter < totalIterations; iter++) {
        anisotropicDiffusion2DKernel<dtype><<<gridSize, blockSize>>>(d_inputImage, d_tempBuffer, deltaT, kappa, diffusionOption, numRows, numCols);

        cudaDeviceSynchronize();  // Synchronous barrier at each time step iteration

        std::swap(d_inputImage, d_tempBuffer);
    }

    cudaMemcpy(inputImage, d_inputImage, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_tempBuffer);
}

/* 
template __global__ void anisotropicDiffusion2DKernel<float>(float*, float*, float, float, int, int, int);
template __global__ void anisotropicDiffusion2DKernel<double>(double*, double*, float, float, int, int, int); */



template  void anisotropicDiffusion2DGPU<float>(float*,   int, float, float, int, int, int);
template  void anisotropicDiffusion2DGPU<double>(double*, int, float, float, int, int, int);

// on device, change name
template<typename dtype>
__global__ void anisotropicDiffusion3DKernel(dtype* inputImage, dtype* outputImage, float deltaT, 
                                           float kappa, int diffusionOption, int numRows, int numCols, int numSlices) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < numRows && col < numCols && depth < numSlices) {

        int idx_center = depth * numRows * numCols + row * numCols + col;

        dtype center = inputImage[idx_center];
        double_t nabla[27];

        int idx_nabla = 0;
        double_t diffusionSum = 0;
        
        for (int dz = -1; dz <= 1; dz++){
            for (int dy = -1; dy <= 1; dy++){
                for (int dx = -1; dx <= 1; dx++){
                
                int currentSlice = depth + dz;
                int currentRow   = row + dy;
                int currentCol   = col + dx;

                //checks for boundaries.
                if (currentRow >= 0 && currentRow < numRows && currentCol >= 0 && currentCol < numCols && currentSlice  >= 0 && currentSlice < numSlices)
                {
                    nabla[idx_nabla] = inputImage[currentSlice * numRows * numCols + currentRow * numCols + currentCol] - center;

                    double scaledDiff = pow(nabla[idx_nabla] / kappa, 2);

                    if (diffusionOption == 1) {
                        diffusionSum += nabla[idx_nabla] * exp(-scaledDiff);
                    } else if (diffusionOption == 2) {
                        diffusionSum += nabla[idx_nabla] / (1 + scaledDiff);
                    } else {
                        diffusionSum+= nabla[idx_nabla] * (1 - tanh(scaledDiff));
                    }
                }
                else
                {
                    nabla[idx_nabla] = 0;
                }

                //update the index of nabla
                idx_nabla++;

                }
            }

        }

        outputImage[idx_center] = inputImage[idx_center] + deltaT * diffusionSum;
    }
}

template<typename dtype>
void anisotropicDiffusion3DGPU(dtype* inputImage, int totalIterations, float deltaT, 
                            float kappa, int diffusionOption, int numRows, int numCols, int numSlices) {
    dtype *d_inputImage, *d_tempBuffer;
    size_t numBytes = numRows * numCols * numSlices * sizeof(dtype);

    // Allocate memory for the input image on the device
    cudaMalloc((void**)&d_inputImage, numBytes);
    cudaMalloc((void**)&d_tempBuffer, numBytes);

    cudaMemcpy(d_inputImage, inputImage, numBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y, (numSlices + blockSize.z - 1) / blockSize.z);


    for (int iter = 0; iter < totalIterations; iter++) {

        anisotropicDiffusion3DKernel<dtype><<<gridSize, blockSize>>>(d_inputImage, d_tempBuffer, deltaT, kappa, diffusionOption, numRows, numCols, numSlices);

        cudaDeviceSynchronize();  // Synchronous barrier at each time step iteration

        std::swap(d_inputImage, d_tempBuffer);
    }

    cudaMemcpy(inputImage, d_inputImage, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_tempBuffer);
}

/* 
template __global__ void anisotropicDiffusion3DKernel<float>(float*, float*,    float, float, int, int, int, int);
template __global__ void anisotropicDiffusion3DKernel<double>(double*, double*, float, float, int, int, int, int);
 */

template  void anisotropicDiffusion3DGPU<float>(float*,    int, float, float, int, int, int, int);
template  void anisotropicDiffusion3DGPU<double>(double*,  int, float, float, int, int, int, int);