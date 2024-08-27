#ifndef COMPLEMENT_BINARY_H
#define COMPLEMENT_BINARY_H

//kernel for image complement_binary
template <typename dtype>
void complement_binary(dtype* deviceImage, dtype* deviceOutput, const int size,
                       const int flag_verbose);

// Slide kernel and erosion operation over all input image pixels
template <typename dtype>
void complement_binary_on_device(dtype* hostImage, dtype* hostOutput, const int size,
                                 const int flag_verbose);

// Slide kernel and erosion operation over all input image pixels
template <typename dtype>
void complement_binary_on_host(dtype* hostImage, dtype* hostOutput, const int size);

#endif  // COMPLEMENT_BINARY_H