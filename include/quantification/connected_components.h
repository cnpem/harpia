#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for 2D label initialization.
 *
 * Assigns initial labels to each foreground pixel in the image.
 *
 * @param[out] block_labels Label buffer for image blocks.
 * @param[in] label_step Step size for accessing the label buffer.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 */
__global__ void Initialization2D(int* block_labels, int label_step, int xsize, int ysize);

/**
 * @brief CUDA kernel to merge equivalent labels in 2D.
 *
 * Uses a union-find structure to unify neighboring labels.
 *
 * @param[in] image Input binary image.
 * @param[in,out] block_labels Label buffer to be merged.
 * @param[in] image_step Step size for accessing the image.
 * @param[in] label_step Step size for accessing the label buffer.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 */
__global__ void Merge2D(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize);

/**
 * @brief CUDA kernel for compressing labels in 2D.
 *
 * Applies path compression to the union-find structure to flatten label trees.
 *
 * @param[in,out] block_labels Label buffer with disjoint set labels.
 * @param[in] label_step Step size for accessing the label buffer.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 */
__global__ void CompressionLabels2D(int* block_labels, int label_step, int xsize, int ysize);

/**
 * @brief CUDA kernel for finalizing the 2D labeling.
 *
 * Assigns the final labels to each pixel after merging and compressing.
 *
 * @param[out] image Output labeled image.
 * @param[in] block_labels Final label values.
 * @param[in] image_step Step size for the output image.
 * @param[in] label_step Step size for label buffer.
 * @param[in] xsize Width of the image.
 * @param[in] ysize Height of the image.
 */
__global__ void FinalLabeling2D(int* image, int* block_labels, int image_step, int label_step, int xsize, int ysize);

/**
 * @brief CUDA kernel for 3D label initialization.
 *
 * Assigns initial labels to each voxel in the 3D volume.
 *
 * @param[out] block_labels 3D label buffer.
 * @param[in] label_step Step size in X-direction for labels.
 * @param[in] ystep Step size in Y-direction.
 * @param[in] zstep Step size in Z-direction.
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
__global__ void Initialization3D(int* block_labels, int label_step, int ystep, int zstep,
                                 int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel to merge equivalent labels in 3D.
 *
 * Merges connected voxels in 3D using 6/26-connectivity.
 *
 * @param[in] image Input 3D binary volume.
 * @param[in,out] block_labels Label buffer to be merged.
 * @param[in] image_step Step size in X-direction for the image.
 * @param[in] label_step Step size in X-direction for labels.
 * @param[in] ystep Step size in Y-direction.
 * @param[in] zstep Step size in Z-direction.
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
__global__ void Merge3D(int* image, int* block_labels, int image_step, int label_step,
                        int ystep, int zstep, int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel for compressing labels in 3D.
 *
 * Compresses paths in the disjoint-set forest used for labeling.
 *
 * @param[in,out] block_labels Label buffer with disjoint set labels.
 * @param[in] label_step Step size in X-direction.
 * @param[in] ystep Step size in Y-direction.
 * @param[in] zstep Step size in Z-direction.
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
__global__ void CompressionLabels3D(int* block_labels, int label_step, int ystep, int zstep,
                                    int xsize, int ysize, int zsize);

/**
 * @brief CUDA kernel for finalizing the 3D labeling.
 *
 * Assigns compressed labels to each voxel.
 *
 * @param[out] image Output labeled 3D volume.
 * @param[in] block_labels Final label values.
 * @param[in] image_step Step size in X-direction for the image.
 * @param[in] label_step Step size in X-direction for labels.
 * @param[in] ystep Step size in Y-direction.
 * @param[in] zstep Step size in Z-direction.
 * @param[in] xsize Width of the volume.
 * @param[in] ysize Height of the volume.
 * @param[in] zsize Depth of the volume.
 */
__global__ void FinalLabeling3D(int* image, int* block_labels, int image_step, int label_step,
                                int ystep, int zstep, int xsize, int ysize, int zsize);

/**
 * @brief Host wrapper for connected components labeling in 2D or 3D.
 *
 * Launches appropriate kernels for label initialization, merging, compression, and finalization.
 *
 * @param[in] image Pointer to input binary image/volume.
 * @param[out] output Pointer to output labeled image/volume.
 * @param[in] xsize Width of the image/volume.
 * @param[in] ysize Height of the image/volume.
 * @param[in] zsize Depth of the volume. Set to 1 for 2D images.
 * @param[in] type If false, apply 2D labeling; if true, apply 3D labeling.
 */
void connectedComponents(int* image, int* output, int xsize, int ysize, int zsize, bool type);

#endif // CONNECTED_COMPONENTS_H