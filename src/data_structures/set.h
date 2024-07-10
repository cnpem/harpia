#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Disjoint set data structure with array-based implementation.
 */

/**
 * Find the root of the set containing element 'a'.
 * @param set Pointer to the disjoint set array.
 * @param a The element to find.
 * @return The root of the set containing 'a'.
 */
__device__  __host__ int find(int* set, int a);

/**
 * Compress the path for the element 'a'.
 * @param set Pointer to the disjoint set array.
 * @param a The element to compress the path for.
 */
__device__ __host__ void compress(int* set, int a);

/**
 * Inline path compression for the element 'a'.
 * @param set Pointer to the disjoint set array.
 * @param a The element to compress the path for inline.
 */
__device__ __host__ void inline_Compress(int* set, int a);

/**
 * Union operation for the CPU.
 * @param set Pointer to the disjoint set array.
 * @param a The first element to union.
 * @param b The second element to union.
 */
__host__ void union_cpu(int* set, int a, int b);

/**
 * Union operation for the GPU.
 * @param set Pointer to the disjoint set array.
 * @param a The first element to union.
 * @param b The second element to union.
 */
__device__ void union_gpu(int* set, int a, int b);
