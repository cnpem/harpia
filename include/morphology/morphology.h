#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_GLOBAL
#endif

typedef enum {
    EROSION,
    DILATION
} MorphOp;

// Define the enum
typedef enum {
    ERODE,
    DILATE,
    TOPHAT,
    BOTTOMHAT
} MorphCV;

typedef struct {
    MorphOp operation1;
    MorphOp operation2;
} MorphChain;


#endif // MORPHOLOGY_H