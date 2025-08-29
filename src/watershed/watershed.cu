#include "../../include/watershed/watershed.h"
#include"../../include/common/union_find.h"
#include <cstdlib>
#include <algorithm>
#include<iostream>
#include<chrono>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <cstdio>
#include <cstring>
#ifdef _OPENMP
  #include <omp.h>
#endif
void get_8_neighbors(int* mask, int i, int j, int rows, int cols)
{
    /*
            mask format
            0   1   2
            3   x   4
            5   6   7
    */   
    
    mask[0] = (i-1)*cols + (j-1);
    mask[1] = (i-1)*cols + (j);
    mask[2] = (i-1)*cols + (j+1);
    mask[3] = (i)*cols + (j-1);
    mask[4] = (i)*cols + (j+1);
    mask[5] = (i+1)*cols + (j-1);
    mask[6] = (i+1)*cols + (j);
    mask[7] = (i+1)*cols + (j+1);


    //test if the neighbors are out of bounds and if they are, set then to -1.
    if (i == 0)
    {
        mask[0] = mask[1] = mask[2] = -1; // Top row
    }

    if (i == rows - 1)
    {
        mask[5] = mask[6] = mask[7] = -1; // Bottom row
    }

    if (j == 0)
    {
        mask[0] = mask[3] = mask[5] = -1; // Left column
    }

    if (j == cols - 1)
    {
        mask[2] = mask[4] = mask[7] = -1; // Right column
    }

}

void get_4_neighbors(int* mask, int i, int j, int rows, int cols)
{
    /*
            mask format
                0   
            1   x   2
                3 
    */

    mask[0] = (i-1)*cols + (j);
    mask[1] = (i)*cols + (j-1);
    mask[2] = (i)*cols + (j+1);
    mask[3] = (i+1)*cols + (j);


    //test if the neighbors are out of bounds and if they are, set then to -1.
    if (i==0)
    {
        mask[0] = -1; //up
    }

    if (j == 0)
    {
         mask[1] = -1; // Left
    }

    if (j == cols - 1)
    {
        mask[2] = -1; // Right
    }

    if (i == rows - 1) 
    {
        mask[3] = -1; // Down
    }
    
    
}

/* Pixel struct for sorting */
typedef struct {
    int idx;
    int val;
} Pixel;

static int cmpPixel(const void* a, const void* b) {
    const Pixel* A = (const Pixel*)a;
    const Pixel* B = (const Pixel*)b;
    if (A->val < B->val) return -1;
    if (A->val > B->val) return 1;
    if (A->idx < B->idx) return -1;
    if (A->idx > B->idx) return 1;
    return 0;
}

/* neighborhood helpers already in your code: get_8_neighbors / get_4_neighbors */

/* --- Initi (sorted version) --- */
static void initi_sorted(int* data, int* labels, int* states, int* mask,
                         int rows, int cols, int* sorted_idx, int n)
{
    for (int k = 0; k < n; ++k) {
        int p = sorted_idx[k];
        int i = p / cols;
        int j = p % cols;

        int min_neighbor_idx = p;
        int min_value = data[p];

        get_4_neighbors(mask, i, j, rows, cols);

        for (int q = 0; q < 4; ++q) {
            if (mask[q] == -1) continue;
            int nc = mask[q];
            int v = data[nc];

            /* pick strictly smaller neighbor OR on tie prefer the larger index
               (this emulates the "max among equals" tie-breaker used in the paper) */
            if (v < min_value || (v == min_value && nc > min_neighbor_idx)) {
                min_value = v;
                min_neighbor_idx = nc;
            }
        }

        if (min_value < data[p]) {
            labels[p] = min_neighbor_idx;
            states[p] = 0;  // downhill
        }
        else if (min_value > data[p]) {
            labels[p] = p;
            states[p] = 1;  // local minima
        }
        else {
            if (min_neighbor_idx > p) {
                labels[p] = min_neighbor_idx;
                states[p] = 2;  // plateau (borrow neighbor)
            } else {
                labels[p] = p;
                states[p] = 3;  // plateau self
            }
        }
    }
}

/* --- Plateau resolution (sorted-aware, correct change logic) --- */
static void plateau_sorted(int* data, int* labels, int* states, int* mask,
                           int rows, int cols, int* sorted_idx, int n)
{
    bool change = true;
    while (change) {
        change = false;
        for (int k = 0; k < n; ++k) {
            int p = sorted_idx[k];
            if (states[p] < 2) continue;  // not a plateau pixel

            int i = p / cols;
            int j = p % cols;
            get_4_neighbors(mask, i, j, rows, cols);

            for (int q = 0; q < 4; ++q) {
                if (mask[q] == -1) continue;
                int nb = mask[q];
                /* only consider downhill neighbors (state==0) with same value */
                if (states[nb] == 0 && data[nb] == data[p]) {
                    labels[p] = nb;
                    states[p] = 0;
                    change = true;
                    break;
                }
            }
        }
    }
}

/* --- Propagation (same as your existing logic, RR==iterations param) --- */
static void propagation(int* labels, int rows, int cols, int RR)
{
    bool change = true;
    //we are computing n a bunch of times... this is bad
    int n = rows * cols;
    while (change) {
        change = false;
        for (int p = 0; p < n; ++p) {
            /* try up to RR shortcuts */
            for (int k = 0; k < RR && labels[p] != labels[labels[p]]; ++k) {
                labels[p] = labels[labels[p]];
                change = true;
            }
        }
    }
}

/* --- Merge (fix bounds check ordering) --- */
static void merge_labels(int* labels, int* states, int* mask, int rows, int cols)
{
    //we are computing n a bunch of times... this is bad
    int n = rows * cols;
    for (int p = 0; p < n; ++p) {
        if (states[p] < 2) continue;  // only minimal plateau pixels
        int i = p / cols;
        int j = p % cols;
        get_4_neighbors(mask, i, j, rows, cols);
        for (int q = 0; q < 4; ++q) {
            if (mask[q] == -1) continue;               // bounds first
            if (states[mask[q]] < 2) continue;         // must also be minimal plateau
            union_cpu(labels, p, mask[q]);             // union sets (uses your union-find)
        }
    }

    /* compress / finalize */
    for (int p = 0; p < n; ++p) {
        inline_Compress(labels, p);
    }
}

/* --- The main watershed (now sorts pixels internally) --- */
void watershed(int* data, int* labels, int rows, int cols, int iterations)
{
    //we are computing n a bunch of times... this is bad
    int n = rows * cols;

    /* prepare sorted pixel indices */
    //unnecessary, remove, because we could just send the original array...
    Pixel* pix = (Pixel*)malloc(n * sizeof(Pixel));
    if (!pix) return;
    for (int i = 0; i < n; ++i) {
        pix[i].idx = i;
        pix[i].val = data[i];
    }
    //need some alternative for cuda?
    // i would maintain this part... but i need to take into account the transfer time in the hierarchical case
    qsort(pix, n, sizeof(Pixel), cmpPixel);

    int* sorted_idx = (int*)malloc(n * sizeof(int));
    if (!sorted_idx) { free(pix); return; }
    for (int i = 0; i < n; ++i) sorted_idx[i] = pix[i].idx;
    free(pix);

    int* states = (int*)malloc(n * sizeof(int));
    if (!states) { free(sorted_idx); return; }

    //in 3d i should use 6 neighbors (its faster this way)
    int* mask = (int*)malloc(4 * sizeof(int));
    if (!mask) { free(states); free(sorted_idx); return; }

    /* Step I: initialize using sorted order */
    initi_sorted(data, labels, states, mask, rows, cols, sorted_idx, n);

    /* Step II: resolve non-minimal plateaux */
    plateau_sorted(data, labels, states, mask, rows, cols, sorted_idx, n);

    /* Step III: propagate / reduce paths */
    propagation(labels, rows, cols, iterations);

    /* Step IV: merge minimal plateau labels */
    merge_labels(labels, states, mask, rows, cols);

    free(mask);
    free(states);
    free(sorted_idx);
}

/* --- Waterfall helpers: Steps V & VI (identify new minima & update image) --- */
static void identify_newmin(int* data, int* labels, int rows, int cols, int* newmin)
{
    //we are computing n a bunch of times... this is bad
    int n = rows * cols;
    /* initialize to sentinel */
    for (int i = 0; i < n; ++i) newmin[i] = INT_MAX;

    int mask[4];
    for (int p = 0; p < n; ++p) {
        int lp = labels[p];
        int i = p / cols;
        int j = p % cols;
        get_4_neighbors(mask, i, j, rows, cols);
        for (int q = 0; q < 4; ++q) {
            if (mask[q] == -1) continue;
            int nb = mask[q];
            if (labels[nb] == lp) continue;
            int h = data[p] > data[nb] ? data[p] : data[nb]; /* max(I(p), I(q)) */
            if (h < newmin[lp]) newmin[lp] = h;
        }
    }
}

static void update_image_with_newmin(int* data, int* labels, int* newmin, int rows, int cols)
{
    //we are computing n a bunch of times... this is bad
    int n = rows * cols;
    for (int p = 0; p < n; ++p) {
        int lp = labels[p];
        if (newmin[lp] != INT_MAX && data[p] < newmin[lp]) {
            data[p] = newmin[lp];
        }
    }
}

/* --- Hierarchical wrapper (keeps the same signature) --- */
void hierarchicalWatershed(int* data, int* labels, int rows, int cols, int levels)
{
    //we are computing n a bunch of times... this is bad
    int n = rows * cols;
    int* newmin = (int*)malloc(n * sizeof(int));
    if (!newmin) {
        /* fallback: run plain watershed */
        //the rr parameter should be 1 in cpu, only in gpu we may need some more iterations for stability
        watershed(data, labels, rows, cols, 1);
        return;
    }

    /* level 0 */
    watershed(data, labels, rows, cols, 1);

    /* subsequent waterfall iterations */
    for (int l = 1; l < levels; ++l) {
        /* Algorithm 4 steps V & VI */
        identify_newmin(data, labels, rows, cols, newmin);
        update_image_with_newmin(data, labels, newmin, rows, cols);

        /* re-run watershed on modified image */
        watershed(data, labels, rows, cols, 1);
    }

    free(newmin);
}


//--------------------------------------------------------------3d----------------------------------------------------//

/* 6-neighbors in 3D */
void get_6_neighbors(int* mask, int x, int y, int z, int rows, int cols, int depth)
{
    /*
        mask format (index order):
        0 = -x
        1 = +x
        2 = -y
        3 = +y
        4 = -z
        5 = +z
    */

    int plane_size = rows * cols;
    int p = z * plane_size + x * cols + y;

    mask[0] = (x > 0)        ? (z * plane_size + (x-1) * cols + y) : -1;
    mask[1] = (x < rows-1)   ? (z * plane_size + (x+1) * cols + y) : -1;
    mask[2] = (y > 0)        ? (z * plane_size + x * cols + (y-1)) : -1;
    mask[3] = (y < cols-1)   ? (z * plane_size + x * cols + (y+1)) : -1;
    mask[4] = (z > 0)        ? ((z-1) * plane_size + x * cols + y) : -1;
    mask[5] = (z < depth-1)  ? ((z+1) * plane_size + x * cols + y) : -1;
}

/* --- Initi (sorted version) --- */
static void initi_sorted_3d(int* data, int* labels, int* states, int* mask,
                            int rows, int cols, int depth, int* sorted_idx, int n)
{
    for (int k = 0; k < n; ++k) {
        int p = sorted_idx[k];
        int z = p / (rows * cols);
        int rem = p % (rows * cols);
        int x = rem / cols;
        int y = rem % cols;

        int min_neighbor_idx = p;
        int min_value = data[p];

        get_6_neighbors(mask, x, y, z, rows, cols, depth);

        for (int q = 0; q < 6; ++q) {
            if (mask[q] == -1) continue;
            int nc = mask[q];
            int v = data[nc];

            if (v < min_value || (v == min_value && nc > min_neighbor_idx)) {
                min_value = v;
                min_neighbor_idx = nc;
            }
        }

        if (min_value < data[p]) {
            labels[p] = min_neighbor_idx;
            states[p] = 0;  // downhill
        }
        else if (min_value > data[p]) {
            labels[p] = p;
            states[p] = 1;  // local minima
        }
        else {
            if (min_neighbor_idx > p) {
                labels[p] = min_neighbor_idx;
                states[p] = 2;  // plateau (borrow neighbor)
            } else {
                labels[p] = p;
                states[p] = 3;  // plateau self
            }
        }
    }
}

/* --- Plateau resolution --- */
static void plateau_sorted_3d(int* data, int* labels, int* states, int* mask,
                              int rows, int cols, int depth, int* sorted_idx, int n)
{
    bool change = true;
    while (change) {
        change = false;
        for (int k = 0; k < n; ++k) {
            int p = sorted_idx[k];
            if (states[p] < 2) continue;

            int z = p / (rows * cols);
            int rem = p % (rows * cols);
            int x = rem / cols;
            int y = rem % cols;

            get_6_neighbors(mask, x, y, z, rows, cols, depth);

            for (int q = 0; q < 6; ++q) {
                if (mask[q] == -1) continue;
                int nb = mask[q];
                if (states[nb] == 0 && data[nb] == data[p]) {
                    labels[p] = nb;
                    states[p] = 0;
                    change = true;
                    break;
                }
            }
        }
    }
}

/* --- Propagation --- */
static void propagation_3d(int* labels, int rows, int cols, int depth, int RR)
{
    bool change = true;
    int n = rows * cols * depth;
    while (change) {
        change = false;
        for (int p = 0; p < n; ++p) {
            for (int k = 0; k < RR && labels[p] != labels[labels[p]]; ++k) {
                labels[p] = labels[labels[p]];
                change = true;
            }
        }
    }
}

/* --- Merge plateaux --- */
static void merge_labels_3d(int* labels, int* states, int* mask,
                            int rows, int cols, int depth)
{
    int n = rows * cols * depth;
    for (int p = 0; p < n; ++p) {
        if (states[p] < 2) continue;
        int z = p / (rows * cols);
        int rem = p % (rows * cols);
        int x = rem / cols;
        int y = rem % cols;

        get_6_neighbors(mask, x, y, z, rows, cols, depth);
        for (int q = 0; q < 6; ++q) {
            if (mask[q] == -1) continue;
            if (states[mask[q]] < 2) continue;
            union_cpu(labels, p, mask[q]);
        }
    }

    for (int p = 0; p < n; ++p) {
        inline_Compress(labels, p);
    }
}

/* --- Main watershed 3D --- */
void watershed3d(int* data, int* labels, int rows, int cols, int depth, int iterations)
{
    int n = rows * cols * depth;

    Pixel* pix = (Pixel*)malloc(n * sizeof(Pixel));
    if (!pix) return;
    for (int i = 0; i < n; ++i) {
        pix[i].idx = i;
        pix[i].val = data[i];
    }
    qsort(pix, n, sizeof(Pixel), cmpPixel);

    int* sorted_idx = (int*)malloc(n * sizeof(int));
    if (!sorted_idx) { free(pix); return; }
    for (int i = 0; i < n; ++i) sorted_idx[i] = pix[i].idx;
    free(pix);

    int* states = (int*)malloc(n * sizeof(int));
    if (!states) { free(sorted_idx); return; }

    int* mask = (int*)malloc(6 * sizeof(int));
    if (!mask) { free(states); free(sorted_idx); return; }

    initi_sorted_3d(data, labels, states, mask, rows, cols, depth, sorted_idx, n);
    plateau_sorted_3d(data, labels, states, mask, rows, cols, depth, sorted_idx, n);
    propagation_3d(labels, rows, cols, depth, iterations);
    merge_labels_3d(labels, states, mask, rows, cols, depth);

    free(mask);
    free(states);
    free(sorted_idx);
}


/* --- Waterfall helpers for 3D: Steps V & VI --- */
static void identify_newmin_3d(int* data, int* labels,
                               int rows, int cols, int depth,
                               int* newmin)
{
    int n = rows * cols * depth;
    for (int i = 0; i < n; ++i) newmin[i] = INT_MAX;

    int mask[6];
    for (int p = 0; p < n; ++p) {
        int lp = labels[p];
        int z = p / (rows * cols);
        int rem = p % (rows * cols);
        int x = rem / cols;
        int y = rem % cols;

        get_6_neighbors(mask, x, y, z, rows, cols, depth);
        for (int q = 0; q < 6; ++q) {
            if (mask[q] == -1) continue;
            int nb = mask[q];
            if (labels[nb] == lp) continue;

            int h = (data[p] > data[nb]) ? data[p] : data[nb];
            if (h < newmin[lp]) newmin[lp] = h;
        }
    }
}

static void update_image_with_newmin_3d(int* data, int* labels,
                                        int* newmin,
                                        int rows, int cols, int depth)
{
    int n = rows * cols * depth;
    for (int p = 0; p < n; ++p) {
        int lp = labels[p];
        if (newmin[lp] != INT_MAX && data[p] < newmin[lp]) {
            data[p] = newmin[lp];
        }
    }
}

/* --- Hierarchical watershed 3D --- */
void hierarchicalWatershed3d(int* data, int* labels,
                             int rows, int cols, int depth,
                             int levels)
{
    int n = rows * cols * depth;
    int* newmin = (int*)malloc(n * sizeof(int));
    if (!newmin) {
        // fallback: run plain watershed3d
        watershed3d(data, labels, rows, cols, depth, 1);
        return;
    }

    // level 0
    watershed3d(data, labels, rows, cols, depth, 1);

    // subsequent waterfall iterations
    for (int l = 1; l < levels; ++l) {
        identify_newmin_3d(data, labels, rows, cols, depth, newmin);
        update_image_with_newmin_3d(data, labels, newmin, rows, cols, depth);
        watershed3d(data, labels, rows, cols, depth, 1);
    }

    free(newmin);
}


//-------------------------------------------batches-------------------------------------------------//
void hierarchicalWatershed_2d_batched(int* image,
                                      int rows, int cols, int depth,
                                      int* labels,
                                      int levels,
                                      int dz)
{
    const int plane = rows * cols;
    const int num_batches = (depth + dz - 1) / dz;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < num_batches; ++b) {
        const int z_start = b * dz;
        const int z_end   = std::min(z_start + dz, depth);

        for (int z = z_start; z < z_end; ++z) {
            int* slice_in  = &image[z * plane];
            int* slice_out = &labels[z * plane];

            // Run hierarchical watershed directly on the slice
            hierarchicalWatershed(slice_in, slice_out, rows, cols, levels);

        }
    }
}