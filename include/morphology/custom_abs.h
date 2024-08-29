#ifndef CUSTOM_ABS_H
#define CUSTOM_ABS_H

#include <cmath>  // For fabs()
#include "morphology.h"

template <typename dtype>
CUDA_HOSTDEV inline dtype custom_abs(dtype value) {
  return (value < 0) ? -value : value;  // Basic abs for general types
}

// Specialization for float types
template <>
CUDA_HOSTDEV inline float custom_abs<float>(float value) {
  return fabsf(value);  // Use fabsf for float
}

// Specialization for int types
template <>
CUDA_HOSTDEV inline int custom_abs<int>(int value) {
  return abs(value);  // Use abs for int
}

// Specialization for unsigned int types
template <>
CUDA_HOSTDEV inline unsigned int custom_abs<unsigned int>(unsigned int value) {
  return value;  // No need for abs on unsigned int since it's always positive
}

#endif  // CUSTOM_ABS_H