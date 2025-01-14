#ifndef LOGICAL_OR_H
#define LOGICAL_OR_H

//kernel for image logical_or
template <typename dtype>
void logical_or(dtype* deviceImage, dtype* deviceOutput, const size_t size,
                       const int flag_verbose);

// Slide kernel and erosion operation over all input image pixels
template <typename dtype>
void logical_or_on_device(dtype* hostImage, dtype* hostOutput, const int xsize,
                                 const int ysize, const int zsize, const int flag_verbose);

// Slide kernel and erosion operation over all input image pixels
template <typename dtype>
void logical_or_on_host(dtype* hostImage, dtype* hostOutput, const size_t size);

#endif  // LOGICAL_OR_H