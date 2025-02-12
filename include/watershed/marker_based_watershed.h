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

void meyers_watershed_2d(int* R, int* M, int bg, int rows, int cols);

void meyers_watershed_3d(int* R, int* M, int bg, int depth, int rows, int cols);

#endif // MARKER_BASED_WATERSHED_H
