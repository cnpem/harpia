#include "../../include/watershed/watershed.h"
#include "../../include/common/union_find.h"
#include "../../include/common/chunkedExecutor.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <climits>

// ------------------------------------------------------------
// Utilities
// ------------------------------------------------------------
// ------------------ GPU compact labels (no Thrust) ------------------
// LUT-based compaction: map each distinct root index -> sequential id 1..K
// Requires deviceLabels to contain root indices (after finalize/inline_Compress).

__global__ void init_lut_kernel(int* lut, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) lut[i] = 0;
}

// Try to claim lut[root] using atomicCAS. If we successfully set lut[root] from 0->-1,
// then we are the first thread to see this root and we allocate an id using atomicAdd(next).
// Finally we write the assigned id into lut[root] (replacing -1).
__global__ void assign_labels_kernel(const int* labels, int* lut, int* next_label, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int root = labels[i];
    // bounds check (safety)
    if (root < 0 || root >= N) return;

    // Try to reserve this lut entry
    int old = atomicCAS(&lut[root], 0, -1);
    if (old == 0) {
        // We are the first: assign a new id (atomicAdd returns old value)
        int id = atomicAdd(next_label, 1); // returns previous value; we use that as id
        // write actual id into lut[root]
        lut[root] = id;
    } else {
        // someone else already reserved/assigned it: do nothing
    }
}

// Remap each label -> lut[label]
__global__ void remap_labels_kernel(int* labels, const int* lut, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int root = labels[i];
        labels[i] = lut[root];
    }
}

// Host wrapper: compact labels on device in-place (labels become 1..K)
void compact_labels_device(int* d_labels, int N)
{
    int *d_lut = nullptr;
    int *d_next = nullptr;

    // allocate
    cudaMalloc(&d_lut, N * sizeof(int));
    cudaMalloc(&d_next, sizeof(int));

    // init lut to 0
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    init_lut_kernel<<<gridSize, blockSize>>>(d_lut, N);
    cudaDeviceSynchronize();

    // initialize next_label = 1 (so first assigned id is 1)
    int init_next = 1;
    cudaMemcpy(d_next, &init_next, sizeof(int), cudaMemcpyHostToDevice);

    // assign labels (may be many collisions, but atomicCAS+atomicAdd serializes per-root assignment)
    assign_labels_kernel<<<gridSize, blockSize>>>(d_labels, d_lut, d_next, N);
    cudaDeviceSynchronize();

    // NOTE: It's possible some lut entries are still 0 for roots that were never present in labels.
    // That's fine: remap accesses only existing roots. Now remap every label -> sequential id
    remap_labels_kernel<<<gridSize, blockSize>>>(d_labels, d_lut, N);
    cudaDeviceSynchronize();

    // free
    cudaFree(d_lut);
    cudaFree(d_next);
}
__host__ __device__ __forceinline__
int index3d(int x, int y, int z, int xsize, int ysize, int /*zsize*/) {
    // layout: (z, y, x) with x fastest
    return (z * ysize + y) * xsize + x;
}

// ------------------------------------------------------------
// INIT (3D): determine min neighbor and set labels/states
// states:
// 0 -> downhill (points to strictly lower-intensity neighbor)
// 1 -> local minimum (root)
// 2 -> plateau borrow (same intensity; adopt larger index neighbor)
// 3 -> plateau self   (same intensity; self-root until merge/resolve)
// ------------------------------------------------------------
template<typename in_dtype>
__global__ void initi_gpu_3d(const in_dtype* deviceImage,
                             int* deviceLabels,
                             int* deviceStates,
                             const int* d_dx, const int* d_dy, const int* d_dz,
                             int nNbrs,
                             int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // x
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; // y
    const int idz = blockIdx.z * blockDim.z + threadIdx.z; // z
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int p = index3d(idx, idy, idz, xsize, ysize, zsize);

    const int orig_val = (int)deviceImage[p];

    int min_val = orig_val;
    int min_idx = p;

    // loop over neighbors (nNbrs = 6 or 26)
    for (int k = 0; k < nNbrs; ++k) {
        const int nx = idx + d_dx[k];
        const int ny = idy + d_dy[k];
        const int nz = idz + d_dz[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= xsize || ny >= ysize || nz >= zsize) continue;

        const int q = index3d(nx, ny, nz, xsize, ysize, zsize);
        const int vq = (int)deviceImage[q];

        // prefer strictly smaller; if tie, prefer larger linear index
        if (vq < min_val || (vq == min_val && q > min_idx)) {
            min_val = vq;
            min_idx = q;
        }
    }

    if (min_val < orig_val) {
        deviceLabels[p] = min_idx; deviceStates[p] = 0;  // downhill
    } else if (min_val > orig_val) {
        deviceLabels[p] = p;       deviceStates[p] = 1;  // strict local minimum
    } else {
        if (min_idx > p) { deviceLabels[p] = min_idx; deviceStates[p] = 2; } // plateau borrow
        else              { deviceLabels[p] = p;       deviceStates[p] = 3; } // plateau self
    }
}

// ------------------------------------------------------------
// PLATEAU RESOLUTION (3D): resolve non-minimal plateau pixels
// Looks for equal-intensity downhill neighbors to borrow
// ------------------------------------------------------------
template<typename in_dtype>
__global__ void plateau_gpu_3d(const in_dtype* deviceImage,
                               int* deviceLabels,
                               int* deviceStates,
                               const int* d_dx, const int* d_dy, const int* d_dz,
                               int nNbrs,
                               int* d_changed,
                               int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int p = index3d(idx, idy, idz, xsize, ysize, zsize);
    const int Sp = deviceStates[p];
    if (Sp < 2) return; // only non-minimal plateau pixels (2 or 3)

    const int Ip = (int)deviceImage[p];

    for (int k = 0; k < nNbrs; ++k) {
        const int nx = idx + d_dx[k];
        const int ny = idy + d_dy[k];
        const int nz = idz + d_dz[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= xsize || ny >= ysize || nz >= zsize) continue;

        const int q = index3d(nx, ny, nz, xsize, ysize, zsize);
        if (deviceStates[q] == 0 && (int)deviceImage[q] == Ip) {
            deviceLabels[p] = q;
            deviceStates[p] = 0;
            atomicExch(d_changed, 1);
            break;
        }
    }
}

// ------------------------------------------------------------
// PROPAGATION (3D): pointer-jumping / path compression (RR steps)
// (unchanged — doesn't use neighbors)
// ------------------------------------------------------------
__global__ void propagation_gpu_3d(int* deviceLabels, int* d_changed,
                                   int xsize, int ysize, int zsize, int RR)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int N = xsize * ysize * zsize;
    const int p = index3d(idx, idy, idz, xsize, ysize, zsize);

    int cur = deviceLabels[p];
    if (cur < 0 || cur >= N) return;

    bool did_change = false;
    for (int r = 0; r < RR; ++r) {
        const int parent = deviceLabels[cur];
        if (parent == cur) break;          // root
        const int newlabel = deviceLabels[parent];
        if (newlabel == deviceLabels[p]) break;
        deviceLabels[p] = newlabel;        // shorten
        cur = deviceLabels[p];
        did_change = true;
    }
    if (did_change) atomicExch(d_changed, 1);
}

// ------------------------------------------------------------
// MERGE (3D): union minimal plateau labels (only q > p to reduce races)
// ------------------------------------------------------------
__global__ void merge_gpu_3d(int* deviceLabels, const int* deviceStates,
                             const int* d_dx, const int* d_dy, const int* d_dz,
                             int nNbrs,
                             int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int p = index3d(idx, idy, idz, xsize, ysize, zsize);
    if (deviceStates[p] < 2) return; // only plateau

    for (int k = 0; k < nNbrs; ++k) {
        const int nx = idx + d_dx[k];
        const int ny = idy + d_dy[k];
        const int nz = idz + d_dz[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= xsize || ny >= ysize || nz >= zsize) continue;

        const int q = index3d(nx, ny, nz, xsize, ysize, zsize);
        if (q <= p) continue;          // reduce duplicate unions
        if (deviceStates[q] < 2) continue;
        union_gpu(deviceLabels, p, q); // your device union
    }
}

// ------------------------------------------------------------
// FINALIZE (3D): compress label pointers (device inline_Compress)
// (unchanged)
// ------------------------------------------------------------
__global__ void finalize_gpu_3d(int* deviceLabels, int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int p = index3d(idx, idy, idz, xsize, ysize, zsize);
    inline_Compress(deviceLabels, p);
}

// ------------------------------------------------------------
// Base watershed: HOST API (3D)
// hostImage shape convention: (zsize, ysize, xsize)
// neighborhood flag: 6 or 27 (we interpret 27 => full 3x3x3 => 26 neighbors)
// ------------------------------------------------------------
template<typename in_dtype>
void watershed_gpu_3d(const in_dtype* hostImage, int* hostLabels,
                      int xsize, int ysize, int zsize,
                      int neighborhood /* 6 or 27 */)
{
    const int N = xsize * ysize * zsize;

    int nNbrs = 6;
    if (neighborhood == 6) nNbrs = 6;
    else if (neighborhood == 27) nNbrs = 26;
    else {
        // fallback, treat any other value as 6-neighborhood
        nNbrs = 6;
    }

    // device memory
    in_dtype* deviceImage = nullptr;
    int *deviceLabels = nullptr, *deviceStates = nullptr;
    int *d_changed = nullptr;
    int *d_dx = nullptr, *d_dy = nullptr, *d_dz = nullptr;

    cudaMalloc(&deviceImage,  N * sizeof(in_dtype));
    cudaMalloc(&deviceLabels, N * sizeof(int));
    cudaMalloc(&deviceStates, N * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_dx, nNbrs * sizeof(int));
    cudaMalloc(&d_dy, nNbrs * sizeof(int));
    cudaMalloc(&d_dz, nNbrs * sizeof(int));

    cudaMemcpy(deviceImage, hostImage, N * sizeof(in_dtype), cudaMemcpyHostToDevice);

    // prepare host offset arrays depending on nNbrs
    std::vector<int> h_dx(nNbrs), h_dy(nNbrs), h_dz(nNbrs);

    if (nNbrs == 6) {
        h_dx = {  1, -1,  0,  0,  0,  0 };
        h_dy = {  0,  0,  1, -1,  0,  0 };
        h_dz = {  0,  0,  0,  0,  1, -1 };
    } else {
        // full 3x3x3 neighborhood excluding center (26 neighbors)
        int idx = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    h_dx[idx] = dx;
                    h_dy[idx] = dy;
                    h_dz[idx] = dz;
                    ++idx;
                }
            }
        }
        // idx should be 26
    }

    cudaMemcpy(d_dx, h_dx.data(), nNbrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy.data(), nNbrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz, h_dz.data(), nNbrs * sizeof(int), cudaMemcpyHostToDevice);

    // launch config
    dim3 block(8, 8, 8);
    dim3 grid((xsize + block.x - 1) / block.x,
              (ysize + block.y - 1) / block.y,
              (zsize + block.z - 1) / block.z);

    // Step 1: init
    initi_gpu_3d<<<grid, block>>>(deviceImage, deviceLabels, deviceStates, d_dx, d_dy, d_dz,
                                  nNbrs, xsize, ysize, zsize);
    cudaDeviceSynchronize();

    // Step 2: plateau resolve (until no change)
    int h_change;
    do {
        h_change = 0;
        cudaMemcpy(d_changed, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        plateau_gpu_3d<<<grid, block>>>(deviceImage, deviceLabels, deviceStates,
                                        d_dx, d_dy, d_dz,
                                        nNbrs,
                                        d_changed, xsize, ysize, zsize);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_change, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_change);

    // Step 3: propagation (pointer jumping) until stable
    do {
        h_change = 0;
        cudaMemcpy(d_changed, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        propagation_gpu_3d<<<grid, block>>>(deviceLabels, d_changed,
                                            xsize, ysize, zsize, /*RR=*/5);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_change, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_change);

    // Step 4: merge plateau components
    merge_gpu_3d<<<grid, block>>>(deviceLabels, deviceStates, d_dx, d_dy, d_dz,
                                  nNbrs, xsize, ysize, zsize);
    cudaDeviceSynchronize();

    // Step 5: finalize (compress)
    finalize_gpu_3d<<<grid, block>>>(deviceLabels, xsize, ysize, zsize);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(hostLabels, deviceLabels, N * sizeof(int), cudaMemcpyDeviceToHost);

    // free
    cudaFree(deviceImage);
    cudaFree(deviceLabels);
    cudaFree(deviceStates);
    cudaFree(d_changed);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dz);
}

// ------------------------------------------------------------
// identify_newmin_gpu_3d: updated to use nNbrs
// ------------------------------------------------------------
template<typename in_dtype>
__global__ void identify_newmin_gpu_3d(const in_dtype* deviceImage,
                                       const int* deviceLabels,
                                       int* d_newmin,
                                       const int* d_dx, const int* d_dy, const int* d_dz,
                                       int nNbrs,
                                       int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int p  = index3d(idx, idy, idz, xsize, ysize, zsize);
    const int lp = deviceLabels[p];
    const int Ip = (int)deviceImage[p];

    for (int k = 0; k < nNbrs; ++k) {
        const int nx = idx + d_dx[k];
        const int ny = idy + d_dy[k];
        const int nz = idz + d_dz[k];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= xsize || ny >= ysize || nz >= zsize) continue;

        const int q  = index3d(nx, ny, nz, xsize, ysize, zsize);
        if (deviceLabels[q] == lp) continue;

        const int Iq = (int)deviceImage[q];
        const int h  = (Ip > Iq) ? Ip : Iq; // max(I(p), I(q))
        atomicMin(&d_newmin[lp], h);
    }
}

// ------------------------------------------------------------
// Update image by new minima (3D) -- unchanged
// ------------------------------------------------------------
template<typename in_dtype>
__global__ void update_image_with_newmin_gpu_3d(in_dtype* deviceImage,
                                                const int* deviceLabels,
                                                const int* d_newmin,
                                                int xsize, int ysize, int zsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= xsize || idy >= ysize || idz >= zsize) return;

    const int p  = index3d(idx, idy, idz, xsize, ysize, zsize);
    const int lp = deviceLabels[p];
    const int nm = d_newmin[lp];

    if (nm != INT_MAX && (int)deviceImage[p] < nm) {
        deviceImage[p] = static_cast<in_dtype>(nm);
    }
}

// ------------------------------------------------------------
// Device-only watershed pass (3D)
// ------------------------------------------------------------
template<typename in_dtype>
void watershed_device_3d(const in_dtype* deviceImage,
                         int* deviceLabels,
                         int* deviceStates,
                         int* d_changed,
                         const int* d_dx, const int* d_dy, const int* d_dz,
                         int nNbrs,
                         int xsize, int ysize, int zsize, int N)
{
    dim3 block(8, 8, 8);
    dim3 grid((xsize + block.x - 1) / block.x,
              (ysize + block.y - 1) / block.y,
              (zsize + block.z - 1) / block.z);

    initi_gpu_3d<<<grid, block>>>(deviceImage, deviceLabels, deviceStates, d_dx, d_dy, d_dz,
                                  nNbrs, xsize, ysize, zsize);
    cudaDeviceSynchronize();

    int h_change;
    do {
        h_change = 0;
        cudaMemcpy(d_changed, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        plateau_gpu_3d<<<grid, block>>>(deviceImage, deviceLabels, deviceStates,
                                        d_dx, d_dy, d_dz,
                                        nNbrs,
                                        d_changed, xsize, ysize, zsize);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_change, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_change);

    do {
        h_change = 0;
        cudaMemcpy(d_changed, &h_change, sizeof(int), cudaMemcpyHostToDevice);

        propagation_gpu_3d<<<grid, block>>>(deviceLabels, d_changed,
                                            xsize, ysize, zsize, /*RR=*/5);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_change, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_change);

    merge_gpu_3d<<<grid, block>>>(deviceLabels, deviceStates, d_dx, d_dy, d_dz,
                                  nNbrs, xsize, ysize, zsize);
    cudaDeviceSynchronize();

    finalize_gpu_3d<<<grid, block>>>(deviceLabels, xsize, ysize, zsize);
    cudaDeviceSynchronize();
}

// ------------------------------------------------------------
// Hierarchical watershed (3D) — levels >= 1
// ------------------------------------------------------------
template<typename in_dtype>
void hierarchicalWatershed_gpu_3d(const in_dtype* hostImage, int* hostLabels,
                                  int xsize, int ysize, int zsize, int flag_verbose, int levels,
                                  int neighborhood /* 6 or 27 */)
{
    const int N = xsize * ysize * zsize;

    // device buffers
    in_dtype*  deviceImage = nullptr;
    int* deviceLabels = nullptr;
    int* deviceStates = nullptr;
    int* d_changed    = nullptr;
    int *d_dx=nullptr, *d_dy=nullptr, *d_dz=nullptr;
    int* d_newmin     = nullptr;

    int nNbrs = (neighborhood == 27) ? 26 : 6;

    cudaMalloc(&deviceImage,  N * sizeof(in_dtype));
    cudaMalloc(&deviceLabels, N * sizeof(int));
    cudaMalloc(&deviceStates, N * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMalloc(&d_dx, nNbrs * sizeof(int));
    cudaMalloc(&d_dy, nNbrs * sizeof(int));
    cudaMalloc(&d_dz, nNbrs * sizeof(int));
    cudaMalloc(&d_newmin, N * sizeof(int));

    cudaMemcpy(deviceImage, hostImage, N * sizeof(in_dtype), cudaMemcpyHostToDevice);

    std::vector<int> h_dx(nNbrs), h_dy(nNbrs), h_dz(nNbrs);
    if (nNbrs == 6) {
        h_dx = {  1, -1,  0,  0,  0,  0 };
        h_dy = {  0,  0,  1, -1,  0,  0 };
        h_dz = {  0,  0,  0,  0,  1, -1 };
    } else {
        int idx = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    h_dx[idx] = dx;
                    h_dy[idx] = dy;
                    h_dz[idx] = dz;
                    ++idx;
                }
            }
        }
    }
    cudaMemcpy(d_dx, h_dx.data(), nNbrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy.data(), nNbrs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz, h_dz.data(), nNbrs * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(8, 8, 8);
    dim3 grid((xsize + block.x - 1) / block.x,
              (ysize + block.y - 1) / block.y,
              (zsize + block.z - 1) / block.z);

     if (flag_verbose==1) {
        printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
        printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    }

    // level 0
    watershed_device_3d(deviceImage, deviceLabels, deviceStates, d_changed,
                        d_dx, d_dy, d_dz, nNbrs,
                        xsize, ysize, zsize, N);

    // higher levels
    for (int l = 1; l < levels; ++l) {
        // reset newmin to INT_MAX
        cudaMemset(d_newmin, 0x7F, N * sizeof(int));

        identify_newmin_gpu_3d<<<grid, block>>>(deviceImage, deviceLabels, d_newmin,
                                                d_dx, d_dy, d_dz, nNbrs,
                                                xsize, ysize, zsize);
        cudaDeviceSynchronize();

        update_image_with_newmin_gpu_3d<in_dtype><<<grid, block>>>(deviceImage, deviceLabels,
                                                                   d_newmin,
                                                                   xsize, ysize, zsize);
        cudaDeviceSynchronize();

        watershed_device_3d(deviceImage, deviceLabels, deviceStates, d_changed,
                            d_dx, d_dy, d_dz, nNbrs,
                            xsize, ysize, zsize, N);
    }
    cudaFree(deviceImage);
    cudaFree(deviceStates);
    cudaFree(d_changed);
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_dz);
    cudaFree(d_newmin);

    // compact labels on device -> sequential 1..K
    compact_labels_device(deviceLabels, N);

    cudaMemcpy(hostLabels, deviceLabels, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceLabels);
    
}



// ------------------- explicit instantiations -------------------
// add instantiations for templates that now accept nNbrs where needed
template __global__ void initi_gpu_3d<float>(const float*, int*, int*, const int*, const int*, const int*, int, int, int, int);
template __global__ void initi_gpu_3d<int>(const int*, int*, int*, const int*, const int*, const int*, int, int, int, int);
template __global__ void initi_gpu_3d<unsigned int>(const unsigned int*, int*, int*, const int*, const int*, const int*, int, int, int, int);

template __global__ void plateau_gpu_3d<float>(const float*, int*, int*, const int*, const int*, const int*, int, int*, int, int, int);
template __global__ void plateau_gpu_3d<int>(const int*, int*, int*, const int*, const int*, const int*, int, int*, int, int, int);
template __global__ void plateau_gpu_3d<unsigned int>(const unsigned int*, int*, int*, const int*, const int*, const int*, int, int*, int, int, int);

// identify_newmin instantiations
template __global__ void identify_newmin_gpu_3d<float>(const float*, const int*, int*, const int*, const int*, const int*, int, int, int, int);
template __global__ void identify_newmin_gpu_3d<int>(const int*, const int*, int*, const int*, const int*, const int*, int, int, int, int);
template __global__ void identify_newmin_gpu_3d<unsigned int>(const unsigned int*, const int*, int*, const int*, const int*, const int*, int, int, int, int);

// update image instantiations (unchanged signatures)
template __global__ void update_image_with_newmin_gpu_3d<float>(float*, const int*, const int*, int, int, int);
template __global__ void update_image_with_newmin_gpu_3d<int>(int*, const int*, const int*, int, int, int);
template __global__ void update_image_with_newmin_gpu_3d<unsigned int>(unsigned int*, const int*, const int*, int, int, int);

// device-level and host-level templates
template void watershed_gpu_3d<float>(const float*, int*, int, int, int, int);
template void watershed_gpu_3d<int>(const int*, int*, int, int, int, int);
template void watershed_gpu_3d<unsigned int>(const unsigned int*, int*, int, int, int, int);

template void watershed_device_3d<float>(const float*, int*, int*, int*, const int*, const int*, const int*, int, int, int, int, int);
template void watershed_device_3d<int>(const int*, int*, int*, int*, const int*, const int*, const int*, int, int, int, int, int);
template void watershed_device_3d<unsigned int>(const unsigned int*, int*, int*, int*, const int*, const int*, const int*, int, int, int, int, int);

template void hierarchicalWatershed_gpu_3d<float>(const float*, int*, int, int, int, int, int,int);
template void hierarchicalWatershed_gpu_3d<int>(const int*, int*, int, int, int, int, int,int);
template void hierarchicalWatershed_gpu_3d<unsigned int>(const unsigned int*, int*, int, int, int,int, int, int);



template<typename in_dtype>
void hierarchicalWatershedChunked(in_dtype* hostImage, int* hostLabels,
                                  int xsize, int ysize, int zsize,
                                  int levels, int neighborhood,
                                  float gpuMemory, int ngpus, int flag_verbose)
{
    if(ngpus == 0)
    {
        throw std::runtime_error("CPU implementation is not available. Ensure a GPU is available.");
    }

    else
    {
        int ncopies = 3;  // same as meanFilterChunked
        chunkedExecutorWatershed(hierarchicalWatershed_gpu_3d<in_dtype>, ncopies, gpuMemory, ngpus,
                        hostImage, hostLabels, xsize, ysize, zsize,
                        flag_verbose, levels, neighborhood);
    }
}


// ------------------------------------------------------------
// Template instantiations for hierarchicalWatershedChunked
// ------------------------------------------------------------
template void hierarchicalWatershedChunked<float>(float* hostImage, int* hostLabels,
                                                  int xsize, int ysize, int zsize,
                                                  int levels, int neighborhood,
                                                  float gpuMemory, int ngpus, int flag_verbose);

template void hierarchicalWatershedChunked<int>(int* hostImage, int* hostLabels,
                                                int xsize, int ysize, int zsize,
                                                int levels, int neighborhood,
                                                float gpuMemory, int ngpus, int flag_verbose);

template void hierarchicalWatershedChunked<unsigned int>(unsigned int* hostImage, int* hostLabels,
                                                         int xsize, int ysize, int zsize,
                                                         int levels, int neighborhood,
                                                         float gpuMemory, int ngpus, int flag_verbose);