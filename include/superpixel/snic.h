#ifndef SNIC
#define SNIC


void snic_grayscale_heap(const float* image, int width, int height,
                         float spacing, int* labels, float m);

void snic_grayscale_heap_2d_batched(const float* image, int width, int height, int depth,
                                    float spacing, int* labels, float m, int dz);

void snic_grayscale_heap_3d(const float* image, int width, int height, int depth,
                            float spacing, int* labels, float m);

void snic_grayscale_heap_3d_batched(const float* image, int width, int height, int depth,
                                    float spacing, int* labels, float m, int dz);


#endif  // SNIC