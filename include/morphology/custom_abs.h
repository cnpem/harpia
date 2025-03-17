#ifndef CUSTOM_ABS_H
#define CUSTOM_ABS_H

#include <cmath>  // For fabs()
#include "morphology.h"

/**
 * @brief Computes the absolute value of a given number.
 *
 * This is a generic implementation for various data types.
 *
 * @tparam dtype Data type of the input value.
 * @param value Input number.
 * @return Absolute value of the input.
 */
template <typename dtype>
CUDA_HOSTDEV inline dtype custom_abs(dtype value) {
  return (value < 0) ? -value : value;  // Basic abs for general types
}

/**
 * @brief Specialization of custom_abs for float.
 *
 * Uses `fabsf()` for improved performance on floating-point values.
 *
 * @param value Input float number.
 * @return Absolute value of the input.
 */
template <>
CUDA_HOSTDEV inline float custom_abs<float>(float value) {
  return fabsf(value);  // Use fabsf for float
}

/**
 * @brief Specialization of custom_abs for int.
 *
 * Uses `abs()` for integer values.
 *
 * @param value Input integer number.
 * @return Absolute value of the input.
 */
template <>
CUDA_HOSTDEV inline int custom_abs<int>(int value) {
  return abs(value);  // Use abs for int
}

/**
 * @brief Specialization of custom_abs for unsigned int.
 *
 * Since unsigned integers are always non-negative, this function 
 * simply returns the input value.
 *
 * @param value Input unsigned integer.
 * @return The same input value.
 */
template <>
CUDA_HOSTDEV inline unsigned int custom_abs<unsigned int>(unsigned int value) {
  return value;  // No need for abs on unsigned int since it's always positive
}

#endif  // CUSTOM_ABS_H