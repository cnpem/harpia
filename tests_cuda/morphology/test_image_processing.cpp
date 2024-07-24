#include "../../include/morphology/test_image_processing.h"
#include "../../include/morphology/morphology.h"

#include <fstream>

// This function show images of any format using opencv. The Normalization step allow to visualize different data formats
// But it also always maximize the contrast in the image, which can cause some distortios.
template<typename dtype>
void showImage2D(dtype *hostImage, const int xsize, const int ysize, const std::string title){

    int size = xsize*ysize;
    
    // Find max value for normalization
    int max = 0;
    for(int i = 0; i<size; i++){
        if(hostImage[i] > max){
            max = hostImage[i];
        }
    }

    // Normalize the slice and convert data to uint8_t
    uint8_t* data = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(hostImage[i]*255/max); 
    }

    // Create a Mat object with the appropriate size, type and data
    cv::Mat image(ysize, xsize, CV_8U, data);

    // Normalize the slice to the range [0, 255] for display purposes (better visualization)
    cv::Mat normalizedImage;
    cv::normalize(image, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display the original image
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::imshow(title, normalizedImage);
    //cv::waitKey(0);

    // Free the allocated memory
    delete[] data;
}
template void showImage2D<int>(int *, const int , const int , const std::string);
template void showImage2D<uint16_t>(uint16_t *, const int , const int , const std::string);
template void showImage2D<float>(float *, const int , const int , const std::string);

template<typename dtype>
void showImage3D(dtype *hostImage, const int xsize, const int ysize, int zsize, const std::string title){
    
    dtype *himg = hostImage;    
    
    for(int slice=0; slice < zsize; slice++){
        showImage2D(himg, xsize, ysize, title+" - slice "+std::to_string(slice));
        himg += xsize*ysize;
    }
}
template void showImage3D<int>(int *, const int , const int , int , const std::string);
template void showImage3D<float>(float *, const int , const int , int , const std::string);


// Implement the erosion operation with OpenCV for 2D images
template<typename dtype, typename dtype2>
void morphology2DopenCV(dtype *hostImage, dtype *hostOutput, 
                   const int kernel_xsize, const int kernel_ysize, 
                   const int xsize, const int ysize, dtype2 operation) {
    
    // Create a cv::Mat object with the appropriate size and type
    cv::Mat image(ysize, xsize, CV_32F);  // Assuming dtype is float, use CV_8U for uchar
    memcpy(image.data, hostImage, xsize * ysize * sizeof(dtype));

    // Create a structuring element
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, 
                                                cv::Size(kernel_xsize, kernel_ysize), 
                                                cv::Point(kernel_xsize / 2, kernel_ysize / 2));
    // Create an output Mat    
    cv::Mat outImage(ysize, xsize, CV_32F);  // Make sure the type matches the image data
    
    switch (operation)
    {
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

    // Copy the result back to hostOutput
    memcpy(hostOutput, outImage.data, xsize * ysize * sizeof(dtype));
}

// Explicit template instantiation
template void morphology2DopenCV<int, MorphCV>(int*, int*, const int, const int, const int, const int, MorphCV);
template void morphology2DopenCV<int, MorphOp>(int*, int*, const int, const int, const int, const int, MorphOp);
template void morphology2DopenCV<float, MorphCV>(float*, float*, const int, const int, const int, const int, MorphCV);
template void morphology2DopenCV<float, MorphOp>(float*, float*, const int, const int, const int, const int, MorphOp);


// Implement the erosion operation with opencv for 3D images and save the result in hostOutout
template<typename dtype, typename dtype2>
void morphology3DopenCV(dtype *hostImage, dtype *hostOutput, 
                   const int kernel_xsize, const int kernel_ysize,
                   const int xsize, const int ysize, const int zsize, dtype2 operation)
{                   
    dtype *himg = hostImage;
    dtype *hout = hostOutput;

    for(int iz=0; iz<zsize; iz++){
        morphology2DopenCV(himg, hout, kernel_xsize, kernel_ysize, xsize, ysize, operation);
        himg += xsize*ysize;
        hout += xsize*ysize;
    }

}
template  void morphology3DopenCV<int, MorphCV>(int*, int*, const int, const int, const int, const int, const int, MorphCV);
template  void morphology3DopenCV<int, MorphOp>(int*,  int*, const int, const int, const int, const int, const int, MorphOp);
template  void morphology3DopenCV<float, MorphCV>(float*, float*, const int, const int, const int, const int, const int, MorphCV);
template  void morphology3DopenCV<float, MorphOp>(float*, float*, const int, const int, const int, const int, const int, MorphOp);