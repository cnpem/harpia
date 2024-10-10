#include "../../include/tests/morphology/test_image_processing.h"
#include "../../include/morphology/morphology.h"

#include <fstream>

/**
 * @brief Displays a 2D image using OpenCV with normalization.
 * 
 * This function normalizes the input image data to fit the 0-255 range for visualization,
 * and then uses OpenCV to display the image. The normalization maximizes contrast, which 
 * may cause some distortion if the original data range is very different.
 * 
 * @tparam dtype The data type of the image elements (e.g., int, uint16_t, float).
 * @param hostImage Pointer to the input image data.
 * @param xsize The width of the image.
 * @param ysize The height of the image.
 * @param title The title for the display window.
 */
template <typename dtype>
void show_image_2D(dtype* hostImage, const int xsize, const int ysize, const std::string title) {
  int size = xsize * ysize;

  // Find the maximum value in the image for normalization
  // max is set for at least 1 to avoid floating point exception
  dtype max = hostImage[0];
  dtype min = hostImage[0];
  for (int i = 1; i < size; i++) {
    if (hostImage[i] > max) {
      max = hostImage[i];
    } else if (hostImage[i] < min) {
      min = hostImage[i];
    }
  }

  if ((max - min) == 0) {
    max += 1;  //avoid zero division
  }

  // Normalize the image data to range [0, 255] and convert to uint8_t
  uint8_t* data = new uint8_t[size];
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>((hostImage[i] - min) * 255 / (max - min));
  }

  // Create a cv::Mat object for OpenCV
  cv::Mat image(ysize, xsize, CV_8U, data);

  // Normalize the image for better visualization
  cv::Mat normalizedImage;
  cv::normalize(image, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);

  // Display the image
  cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
  cv::imshow(title, normalizedImage);

  // Free the allocated memory
  delete[] data;
}

// Explicit template instantiations for different data types
template void show_image_2D<float>(float*, const int, const int, const std::string);
template void show_image_2D<int>(int*, const int, const int, const std::string);
template void show_image_2D<unsigned int>(unsigned int*, const int, const int, const std::string);
template void show_image_2D<int16_t>(int16_t*, const int, const int, const std::string);
template void show_image_2D<uint16_t>(uint16_t*, const int, const int, const std::string);
template void show_image_2D<int8_t>(int8_t*, const int, const int, const std::string);
template void show_image_2D<uint8_t>(uint8_t*, const int, const int, const std::string);

/**
 * @brief Displays a 3D image by showing each slice as a 2D image.
 * 
 * This function iterates over the slices of a 3D image and calls `show_image_2D` to display 
 * each slice individually.
 * 
 * @tparam dtype The data type of the image elements (e.g., int, float).
 * @param hostImage Pointer to the input 3D image data.
 * @param xsize The width of each slice.
 * @param ysize The height of each slice.
 * @param zsize The number of slices (depth) in the 3D image.
 * @param title The title prefix for the display windows.
 */
template <typename dtype>
void show_image_3D(dtype* hostImage, const int xsize, const int ysize, int zsize,
                   const std::string title) {
  dtype* himg = hostImage;
  if (zsize > 10) {
    printf("Too many windows to plot. Cannot plot %d windows.\n", zsize);
    return;
  }
  // Display each slice of the 3D image
  for (int slice = 0; slice < zsize; slice++) {
    show_image_2D(himg, xsize, ysize, title + " - slice " + std::to_string(slice));
    himg += xsize * ysize;
  }
}

// Explicit template instantiations for different data types
template void show_image_3D<float>(float*, const int, const int, int, const std::string);
template void show_image_3D<int>(int*, const int, const int, int, const std::string);
template void show_image_3D<unsigned int>(unsigned int*, const int, const int, int,
                                          const std::string);
template void show_image_3D<int16_t>(int16_t*, const int, const int, int, const std::string);
template void show_image_3D<uint16_t>(uint16_t*, const int, const int, int, const std::string);
template void show_image_3D<int8_t>(int8_t*, const int, const int, int, const std::string);
template void show_image_3D<uint8_t>(uint8_t*, const int, const int, int, const std::string);

/**
 * @brief Performs morphological operations on a 2D image using OpenCV.
 * 
 * This function applies a specified morphological operation (e.g., erosion, dilation) 
 * to a 2D image using OpenCV's built-in functions.
 * 
 * @tparam dtype The data type of the image elements (e.g., int, float).
 * @tparam dtype2 The type of the morphological operation enum.
 * @param hostImage Pointer to the input image data.
 * @param hostOutput Pointer to the output image data.
 * @param kernel_xsize The width of the structuring element.
 * @param kernel_ysize The height of the structuring element.
 * @param xsize The width of the image.
 * @param ysize The height of the image.
 * @param operation The morphological operation to apply.
 */
template <typename dtype, typename dtype2>
void morphology_2D_openCV(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int kernel_xsize, const int kernel_ysize, dtype2 operation) {
  // Create a cv::Mat object for the input image
  cv::Mat image(ysize, xsize, CV_32F);  // Use CV_32F for float data
  memcpy(image.data, hostImage, xsize * ysize * sizeof(dtype));

  // Create a structuring element
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_xsize, kernel_ysize),
                                              cv::Point(kernel_xsize / 2, kernel_ysize / 2));
  // Create an output Mat
  cv::Mat outImage(ysize, xsize,
                   CV_32F);  // Output image should match the input type

  // Apply the selected morphological operation
  switch (operation) {
    case ERODE:
      cv::erode(image, outImage, element);
      break;
    case DILATE:
      cv::dilate(image, outImage, element);
      break;
    case TOPHAT:
      cv::morphologyEx(image, outImage, cv::MORPH_TOPHAT, element);
      break;
    case BOTTOMHAT:
      cv::morphologyEx(image, outImage, cv::MORPH_BLACKHAT, element);
      break;
    default:
      break;
  }

  // Copy the result back to the output buffer
  memcpy(hostOutput, outImage.data, xsize * ysize * sizeof(dtype));
}

// Explicit template instantiations for different data types and operations
template void morphology_2D_openCV<int, MorphCV>(int*, int*, const int, const int, const int,
                                                 const int, MorphCV);
template void morphology_2D_openCV<int, MorphOp>(int*, int*, const int, const int, const int,
                                                 const int, MorphOp);
template void morphology_2D_openCV<float, MorphCV>(float*, float*, const int, const int, const int,
                                                   const int, MorphCV);
template void morphology_2D_openCV<float, MorphOp>(float*, float*, const int, const int, const int,
                                                   const int, MorphOp);

/**
 * @brief Performs morphological operations on a 3D image using OpenCV.
 * 
 * This function applies the specified morphological operation to each slice of a 3D image.
 * 
 * @tparam dtype The data type of the image elements (e.g., int, float).
 * @tparam dtype2 The type of the morphological operation enum.
 * @param hostImage Pointer to the input 3D image data.
 * @param hostOutput Pointer to the output 3D image data.
 * @param kernel_xsize The width of the structuring element.
 * @param kernel_ysize The height of the structuring element.
 * @param xsize The width of each slice.
 * @param ysize The height of each slice.
 * @param zsize The number of slices (depth) in the 3D image.
 * @param operation The morphological operation to apply.
 */
template <typename dtype, typename dtype2>
void morphology_3D_openCV(dtype* hostImage, dtype* hostOutput, const int xsize, const int ysize,
                          const int zsize, const int kernel_xsize, const int kernel_ysize,
                          dtype2 operation) {
  dtype* himg = hostImage;
  dtype* hout = hostOutput;

  // Apply the morphological operation to each slice
  for (int iz = 0; iz < zsize; iz++) {
    morphology_2D_openCV(himg, hout, xsize, ysize, kernel_xsize, kernel_ysize, operation);
    himg += xsize * ysize;
    hout += xsize * ysize;
  }
}

// Explicit template instantiations for different data types and operations
template void morphology_3D_openCV<int, MorphCV>(int*, int*, const int, const int, const int,
                                                 const int, const int, MorphCV);
template void morphology_3D_openCV<int, MorphOp>(int*, int*, const int, const int, const int,
                                                 const int, const int, MorphOp);
template void morphology_3D_openCV<float, MorphCV>(float*, float*, const int, const int, const int,
                                                   const int, const int, MorphCV);
template void morphology_3D_openCV<float, MorphOp>(float*, float*, const int, const int, const int,
                                                   const int, const int, MorphOp);
