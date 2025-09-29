#include<iostream>
#include"../../include/watershed/marker_based_watershed.h"

/*

    Meyer's watershed algorithm, it uses a priority queue as data structure.

*/

#define MAX_SIZE 9000000 // Define a maximum size for the priority queue

// Define a structure to hold the priority queue and its size
struct  PriorityQueue2d {
    static int PQ[MAX_SIZE][3]; // PQ[i][0]: intensity, PQ[i][1]: x, PQ[i][2]: y
    int size; // Size of the priority queue
};

int PriorityQueue2d::PQ[MAX_SIZE][3];

// Function to initialize the priority queue
void init_priority_queue_2d(PriorityQueue2d* pq) {
    pq->size = 0;
}

// Function to insert into the priority queue
void insert_min_heap_2d(PriorityQueue2d* pq, int intensity, int x, int y) {
    if (pq->size >= MAX_SIZE) return; // Handle heap overflow
    pq->PQ[pq->size][0] = intensity;
    pq->PQ[pq->size][1] = x;
    pq->PQ[pq->size][2] = y;
    int idx = pq->size;
    pq->size++;

    // Bubble up
    while (idx > 0 && pq->PQ[(idx - 1) / 2][0] > pq->PQ[idx][0]) {
        // Swap with parent
        int temp[3];
        for (int i = 0; i < 3; i++) {
            temp[i] = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[(idx - 1) / 2][i];
            pq->PQ[(idx - 1) / 2][i] = temp[i];
        }
        idx = (idx - 1) / 2;
    }
}

// Function to extract the minimum from the priority queue
void extract_min_2d(PriorityQueue2d* pq, int* intensity, int* x, int* y) {
    if (pq->size == 0) {
        *intensity = INT_MAX; // Return a dummy value
        *x = -1;
        *y = -1;
        return;
    }
    *intensity = pq->PQ[0][0];
    *x = pq->PQ[0][1];
    *y = pq->PQ[0][2];

    // Move the last element to the root and bubble down
    pq->PQ[0][0] = pq->PQ[pq->size - 1][0];
    pq->PQ[0][1] = pq->PQ[pq->size - 1][1];
    pq->PQ[0][2] = pq->PQ[pq->size - 1][2];
    pq->size--;

    int idx = 0;
    while (1) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < pq->size && pq->PQ[left][0] < pq->PQ[smallest][0])
            smallest = left;
        if (right < pq->size && pq->PQ[right][0] < pq->PQ[smallest][0])
            smallest = right;

        if (smallest == idx) break; // If the smallest is the current index, we are done

        // Swap with the smallest child
        int temp[3];
        for (int i = 0; i < 3; i++) {
            temp[i] = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[smallest][i];
            pq->PQ[smallest][i] = temp[i];
        }
        idx = smallest;
    }
}

// Marker-controlled watershed algorithm
void meyers_watershed_2d(int* R, int* M, int bg, int rows, int cols) {
    PriorityQueue2d pq;
    init_priority_queue_2d(&pq); // Initialize the priority queue

    // Define 4-connected neighbors offsets
    int neighborOffsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // Initialize the priority queue with all markers
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (M[i * cols + j] != bg) {
                for (int k = 0; k < 4; k++) {
                    int ni = i + neighborOffsets[k][0];
                    int nj = j + neighborOffsets[k][1];
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && M[ni * cols + nj] == bg) {
                        insert_min_heap_2d(&pq, R[i * cols + j], i, j);
                        break;
                    }
                }
            }
        }
    }

    // Perform the watershed algorithm
    while (pq.size > 0) {
        int intensity, i, j;
        extract_min_2d(&pq, &intensity, &i, &j);

        // Iterate over the neighbors of the current pixel
        for (int k = 0; k < 4; k++) {
            int ni = i + neighborOffsets[k][0];
            int nj = j + neighborOffsets[k][1];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && M[ni * cols + nj] == bg) {
                M[ni * cols + nj] = M[i * cols + j];
                insert_min_heap_2d(&pq, (R[i * cols + j] > R[ni * cols + nj]) ? R[i * cols + j] : R[ni * cols + nj], ni, nj);
            }
        }
    }
}

struct PriorityQueue3d {
    static int PQ[MAX_SIZE][4]; // PQ[i][0]: intensity, PQ[i][1]: x, PQ[i][2]: y, PQ[i][3]: z
    int size; // Size of the priority queue
};


int PriorityQueue3d::PQ[MAX_SIZE][4];

// Function to initialize the priority queue
void init_priority_queue_3d(PriorityQueue3d* pq) {
    pq->size = 0;
}

// Function to insert into the priority queue
void insert_min_heap_3d(PriorityQueue3d* pq, int intensity, int x, int y, int z) {
    if (pq->size >= MAX_SIZE) return; // Handle heap overflow
    pq->PQ[pq->size][0] = intensity;
    pq->PQ[pq->size][1] = x;
    pq->PQ[pq->size][2] = y;
    pq->PQ[pq->size][3] = z; // Store z-coordinate
    int idx = pq->size;
    pq->size++;

    // Bubble up
    while (idx > 0 && pq->PQ[(idx - 1) / 2][0] > pq->PQ[idx][0]) {
        // Swap with parent
        int temp[4];
        for (int i = 0; i < 4; i++) {
            temp[i] = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[(idx - 1) / 2][i];
            pq->PQ[(idx - 1) / 2][i] = temp[i];
        }
        idx = (idx - 1) / 2;
    }
}

// Function to extract the minimum from the priority queue
void extract_min_3d(PriorityQueue3d* pq, int* intensity, int* x, int* y, int* z) {
    if (pq->size == 0) {
        *intensity = INT_MAX; // Return a dummy value
        *x = -1;
        *y = -1;
        *z = -1;
        return;
    }
    *intensity = pq->PQ[0][0];
    *x = pq->PQ[0][1];
    *y = pq->PQ[0][2];
    *z = pq->PQ[0][3]; // Retrieve z-coordinate

    // Move the last element to the root and bubble down
    pq->PQ[0][0] = pq->PQ[pq->size - 1][0];
    pq->PQ[0][1] = pq->PQ[pq->size - 1][1];
    pq->PQ[0][2] = pq->PQ[pq->size - 1][2];
    pq->PQ[0][3] = pq->PQ[pq->size - 1][3]; // Move z-coordinate
    pq->size--;

    int idx = 0;
    while (1) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < pq->size && pq->PQ[left][0] < pq->PQ[smallest][0])
            smallest = left;
        if (right < pq->size && pq->PQ[right][0] < pq->PQ[smallest][0])
            smallest = right;

        if (smallest == idx) break; // If the smallest is the current index, we are done

        // Swap with the smallest child
        int temp[4];
        for (int i = 0; i < 4; i++) {
            temp[i] = pq->PQ[idx][i];
            pq->PQ[idx][i] = pq->PQ[smallest][i];
            pq->PQ[smallest][i] = temp[i];
        }
        idx = smallest;
    }
}

// Marker-controlled watershed algorithm
void meyers_watershed_3d(int* R, int* M, int bg, int depth, int rows, int cols) {
    PriorityQueue3d pq;
    init_priority_queue_3d(&pq); // Initialize the priority queue

    // Define 6-connected neighbors offsets (including z-dimension)
    int neighborOffsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, // X neighbors
        {0, -1, 0}, {0, 1, 0}, // Y neighbors
        {0, 0, -1}, {0, 0, 1}  // Z neighbors
    };

    // Initialize the priority queue with all markers
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (M[d * rows * cols + i * cols + j] != bg) {
                    for (int k = 0; k < 6; k++) {
                        int ni = i + neighborOffsets[k][1];
                        int nj = j + neighborOffsets[k][2];
                        int nd = d + neighborOffsets[k][0];
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && nd >= 0 && nd < depth && M[nd * rows * cols + ni * cols + nj] == bg) {
                            insert_min_heap_3d(&pq, R[d * rows * cols + i * cols + j], i, j, d);
                            break;
                        }
                    }
                }
            }
        }
    }

    // Perform the watershed algorithm
    while (pq.size > 0) {
        int intensity, i, j, d;
        extract_min_3d(&pq, &intensity, &i, &j, &d);

        // Iterate over the neighbors of the current pixel
        for (int k = 0; k < 6; k++) {
            int ni = i + neighborOffsets[k][1];
            int nj = j + neighborOffsets[k][2];
            int nd = d + neighborOffsets[k][0];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && nd >= 0 && nd < depth && M[nd * rows * cols + ni * cols + nj] == bg) {
                M[nd * rows * cols + ni * cols + nj] = M[d * rows * cols + i * cols + j];
                insert_min_heap_3d(&pq, (R[d * rows * cols + i * cols + j] > R[nd * rows * cols + ni * cols + nj]) ? R[d * rows * cols + i * cols + j] : R[nd * rows * cols + ni * cols + nj], ni, nj, nd);
            }
        }
    }
}

