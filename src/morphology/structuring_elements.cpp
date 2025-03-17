#include "../../include/morphology/structuring_elements.h"
#include <stdio.h>
#include <stdlib.h>

void get_structuring_element_3D(int* kernel, int kernel_xsize, int kernel_ysize, int kernel_zsize) {
  int size = kernel_xsize * kernel_ysize * kernel_zsize;

  if (!kernel) {
    printf("Failed to define kernel.\n");
    return;
  }

  for (int i = 0; i < size; i++) {
    kernel[i] = 1;
  }
}

void horizontal_line_kernel(int* kernel) {
  if (!kernel) {
    printf("Failed to define kernel.\n");
    return;
  }

  int* ik = kernel;
  for (int i = 0; i < 3; i++) {
    ik[0] = 1;
    ik[1] = 1;
    ik[2] = 1;
    ik[3] = 0;
    ik[4] = 0;
    ik[5] = 0;
    ik[6] = 0;
    ik[7] = 0;
    ik[8] = 0;

    ik += 9;  // Repeat the defined slice along z axis
  }
}

void vertical_line_kernel(int* kernel) {
  if (!kernel) {
    printf("Failed to define kernel.\n");
    return;
  }

  int* ik = kernel;
  for (int i = 0; i < 3; i++) {
    ik[0] = 1;
    ik[1] = 0;
    ik[2] = 0;
    ik[3] = 1;
    ik[4] = 0;
    ik[5] = 0;
    ik[6] = 1;
    ik[7] = 0;
    ik[8] = 0;

    ik += 9;  // Repeat the defined slice along z axis
  }
}

void custum_kernel_3D(int* kernel) {
  if (!kernel) {
    printf("Failed to define kernel.\n");
    return;
  }

  int* ik = kernel;
  for (int i = 0; i < 3; i++) {
    ik[0] = 1;
    ik[1] = 0;
    ik[2] = -1;
    ik[3] = 1;
    ik[4] = 0;
    ik[5] = -1;
    ik[6] = 1;
    ik[7] = 0;
    ik[8] = -1;

    ik += 9;  // Repeat the defined slice along z axis
  }
}