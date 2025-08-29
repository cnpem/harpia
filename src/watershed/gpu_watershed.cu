#include "../../include/watershed/watershed.h"
#include"../../include/common/union_find.h"
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include <algorithm>
#include <vector>

// Sort pixel indices by intensity
template<typename in_dtype>
void build_sorted_index(const in_dtype* h_image, int rows, int cols, int* sorted_idx)
{
    int N = rows * cols;
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return h_image[a] < h_image[b]; });

    for (int i = 0; i < N; ++i) sorted_idx[i] = idx[i];
}
// ---------- corrected kernels ----------

// INIT: determine min neighbor and set labels/states
template<typename in_dtype>
__global__ void initi_gpu(const in_dtype* image, int* labels, int* states,
                          const int* dx, const int* dy,
                          int xsize, int ysize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x >= xsize || y >= ysize) return;

    const int p = y * xsize + x;             // correct linear index
    const int orig_val = (int)image[p];
    int min_val = orig_val;
    int min_idx = p;

    // 4-neighbors using dx/dy arrays
    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx < 0 || ny < 0 || nx >= xsize || ny >= ysize) continue;
        int q = ny * xsize + nx;
        int vq = (int)image[q];
        // prefer strictly smaller, tie-breaker: larger index
        if (vq < min_val || (vq == min_val && q > min_idx)) {
            min_val = vq;
            min_idx = q;
        }
    }

    if (min_val < orig_val) {
        labels[p] = min_idx; states[p] = 0;        // downhill
    } else if (min_val > orig_val) {
        labels[p] = p;       states[p] = 1;        // local minimum
    } else {
        if (min_idx > p) { labels[p] = min_idx; states[p] = 2; } // plateau borrow
        else              { labels[p] = p;       states[p] = 3; } // plateau self
    }
}

// Explicit instantiations
template __global__ void initi_gpu<float>(const float*, int*, int*, const int*, const int*, int, int);
template __global__ void initi_gpu<int>(const int*, int*, int*, const int*, const int*, int, int);
template __global__ void initi_gpu<unsigned int>(const unsigned int*, int*, int*, const int*, const int*, int, int);


// PLATEAU: resolve non-minimal plateau pixels
template<typename in_dtype>
__global__ void plateau_gpu(const in_dtype* image, int* labels, int* states,
                            const int* dx, const int* dy, int* d_changed,
                            int xsize, int ysize )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xsize || y >= ysize) return;

    const int p = y * xsize + x;
    int Sp = states[p];
    if (Sp < 2) return;    // not a non-minimal plateau pixel

    int orig_val = (int)image[p];

    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx < 0 || ny < 0 || nx >= xsize || ny >= ysize) continue;
        int q = ny * xsize + nx;
        if (states[q] == 0 && (int)image[q] == orig_val) {
            labels[p] = q;
            states[p] = 0;
            atomicExch(d_changed, 1);
            break;
        }
    }
}

// Explicit instantiations
template __global__ void plateau_gpu<float>(const float*, int*, int*, const int*, const int*, int*, int, int);
template __global__ void plateau_gpu<int>(const int*, int*, int*, const int*, const int*, int*, int, int);
template __global__ void plateau_gpu<unsigned int>(const unsigned int*, int*, int*, const int*, const int*, int*, int, int);

// PROPAGATION: pointer shortening up to RR steps; set d_changed if any thread changed
__global__ void propagation_gpu(int* labels, int* d_changed, int xsize, int ysize, int RR)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xsize || y >= ysize) return;

    const int n = xsize * ysize;
    const int p = y * xsize + x;

    int cur = labels[p];
    if (cur < 0 || cur >= n) return;

    bool did_change = false;
    for (int r = 0; r < RR; ++r) {
        int parent = labels[cur];
        if (parent == cur) break;            // reached root
        int newlabel = labels[parent];       // safe: parent in [0,n)
        if (newlabel == labels[p]) break;    // no further shortening
        labels[p] = newlabel;                // shorten pointer
        cur = labels[p];
        did_change = true;
    }
    if (did_change) atomicExch(d_changed, 1);
}


// MERGE: union minimal plateau labels (only q > p to reduce race)
__global__ void merge_gpu(int* labels, const int* states, const int* dx, const int* dy,
                          int xsize, int ysize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xsize || y >= ysize) return;

    const int p = y * xsize + x;
    if (states[p] < 2) return;

    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx < 0 || ny < 0 || nx >= xsize || ny >= ysize) continue;
        int q = ny * xsize + nx;
        if (q <= p) continue;
        if (states[q] < 2) continue;
        // device union (you provided union_gpu in union-find file)
        union_gpu(labels, p, q);
    }
}


// FINALIZE: compress all label pointers (device inline_Compress)
__global__ void finalize_gpu(int* labels, int xsize, int ysize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xsize || y >= ysize) return;

    const int p = y * xsize + x;
    inline_Compress(labels, p);
}


template<typename in_dtype>
void watershed_gpu(const in_dtype* h_image, int* h_labels, int rows, int cols)
{
    const int N = rows * cols;

    // --- Allocate device memory ---
    in_dtype* d_image;
    int* d_labels;
    int* d_states;
    int* d_changed;
    int* d_dx;
    int* d_dy;
    int* d_sorted_idx;

    cudaMalloc(&d_image,  N * sizeof(in_dtype));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_states, N * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_dx, 4 * sizeof(int));
    cudaMalloc(&d_dy, 4 * sizeof(int));
    cudaMalloc(&d_sorted_idx, N * sizeof(int));

    // copy image to device
    cudaMemcpy(d_image, h_image, N * sizeof(in_dtype), cudaMemcpyHostToDevice);

    // 4-neighbors
    int h_dx[4] = {1, -1, 0, 0};
    int h_dy[4] = {0, 0, 1, -1};
    cudaMemcpy(d_dx, h_dx, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, 4 * sizeof(int), cudaMemcpyHostToDevice);

    // --- Build sorted index on host ---
    int* h_sorted_idx = new int[N];
    build_sorted_index(h_image, rows, cols, h_sorted_idx);
    cudaMemcpy(d_sorted_idx, h_sorted_idx, N * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_sorted_idx;

    // --- Kernel launch config ---
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    // --- Step 1: Initialization ---
    initi_gpu<<<grid, block>>>(d_image, d_labels, d_states,
                               d_dx, d_dy, cols, rows);
    cudaDeviceSynchronize();

    // --- Step 2: Plateau resolution ---
    int h_change;
    do {
        h_change = 0;
        cudaMemcpy(d_changed, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        plateau_gpu<<<grid, block>>>(d_image, d_labels, d_states,
                                     d_dx, d_sorted_idx, d_changed,
                                     cols, rows);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_change, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_change);

    // --- Step 3: Propagation ---
    do {
        h_change = 0;
        cudaMemcpy(d_changed, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        propagation_gpu<<<grid, block>>>(d_labels, d_changed, cols, rows,150);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_change, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_change);

    // --- Step 4: Merge ---
    merge_gpu<<<grid, block>>>(d_labels, d_states, d_dx, d_dy, cols, rows);
    cudaDeviceSynchronize();

    // --- Step 5: Finalize ---
    finalize_gpu<<<grid, block>>>(d_labels, cols, rows);
    cudaDeviceSynchronize();

    // --- Copy back ---
    cudaMemcpy(h_labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Free ---
    cudaFree(d_image);
    cudaFree(d_labels);
    cudaFree(d_states);
    cudaFree(d_changed);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_sorted_idx);
}

// Explicit instantiations
template void watershed_gpu<float>(const float*, int*, int, int);
template void watershed_gpu<int>(const int*, int*, int, int);
template void watershed_gpu<unsigned int>(const unsigned int*, int*, int, int);