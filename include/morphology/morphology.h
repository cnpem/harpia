/**
 * @file morphology.h
 * @brief Defines morphological operations and structures for GPU and CPU processing.
 *
 * This header file provides definitions for morphological operations used in image 
 * processing. It includes CUDA compatibility macros, enumerations for different 
 * morphology types, and a struct for chaining operations.
 */

 #ifndef MORPHOLOGY_H
 #define MORPHOLOGY_H
 
 #ifdef __CUDACC__
 /**
  * @def CUDA_HOSTDEV
  * @brief Marks a function to be callable from both host and device code in CUDA.
  */
 #define CUDA_HOSTDEV __host__ __device__
 
 /**
  * @def CUDA_GLOBAL
  * @brief Marks a function as a CUDA kernel (callable from host but executed on device).
  */
 #define CUDA_GLOBAL __global__
 #else
 #define CUDA_HOSTDEV
 #define CUDA_GLOBAL
 #endif
 
 /**
  * @enum MorphOp
  * @brief Enum representing basic morphological operations.
  */
 typedef enum {
   EROSION,  ///< Erosion operation
   DILATION  ///< Dilation operation
 } MorphOp;
 
 /**
  * @enum MorphCV
  * @brief Enum representing extended morphological operations.
  */
 typedef enum {
   ERODE,      ///< Erosion operation (same as EROSION)
   DILATE,     ///< Dilation operation (same as DILATION)
   TOPHAT,     ///< Top-hat transformation
   BOTTOMHAT   ///< Bottom-hat transformation
 } MorphCV;
 
 /**
  * @struct MorphChain
  * @brief Structure to define a sequence of two morphological operations.
  */
 typedef struct {
   MorphOp operation1; ///< First morphological operation
   MorphOp operation2; ///< Second morphological operation
 } MorphChain;
 
 #endif  // MORPHOLOGY_H