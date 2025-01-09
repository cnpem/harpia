#ifndef MARKER_BASED_WATERSHED_H
#define MARKER_BASED_WATERSHED_H

#include <iostream>
#include <climits>

#define MAX_SIZE 9000000  // Define a maximum size for the priority queue

// Union-Find operations
int findd(int* set, int x);
void unionn(int* set, int x, int y);
void union_find_markers(int* labels, int idx, int idy, int xsize, int ysize);
void union_find_non_markers(int* labels, int idx, int idy, int xsize, int ysize);
void union_find_watershed(int* sortedImage, int* labels, int xsize, int ysize);

// Priority Queue for 2D Watershed
struct  PriorityQueue2d {
    static int PQ[MAX_SIZE][3]; // PQ[i][0]: intensity, PQ[i][1]: x, PQ[i][2]: y
    int size; // Size of the priority queue
};

int PriorityQueue2d::PQ[MAX_SIZE][3];

void init_priority_queue_2d(PriorityQueue2d* pq);
void insert_min_heap_2d(PriorityQueue2d* pq, int intensity, int x, int y);
void extract_min_2d(PriorityQueue2d* pq, int* intensity, int* x, int* y);
void meyers_watershed_2d(int* R, int* M, int bg, int rows, int cols);

struct  PriorityQueue2d {
    static int PQ[MAX_SIZE][4]; // PQ[i][0]: intensity, PQ[i][1]: x, PQ[i][2]: y
    int size; // Size of the priority queue
};

int PriorityQueue3d::PQ[MAX_SIZE][4];

void init_priority_queue_3d(PriorityQueue3d* pq);
void insert_min_heap_3d(PriorityQueue3d* pq, int intensity, int x, int y, int z);
void extract_min_3d(PriorityQueue3d* pq, int* intensity, int* x, int* y, int* z);
void meyers_watershed_3d(int* R, int* M, int bg, int depth, int rows, int cols);

#endif // MARKER_BASED_WATERSHED_H
