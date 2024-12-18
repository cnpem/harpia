#include<iostream>
#include<cuda_runtime.h>

void watershed(int* data, int* labels, int rows, int cols, int iterations);
