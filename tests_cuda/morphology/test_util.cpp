#include "../../include/morphology/test_util.h"
#include <fstream>
#include <iomanip>
#include <cstdint> // For uint16_t
#include <sys/time.h> // For gettimeofday

double cpu_second(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);//convert usec to sec and sum correctly
}

// Read input raw uint16_t image data to dtype allocated memory pointer
template<typename dtype>
void read_input(dtype *image, const std::string& filename, const int size, const int flag_verbose){

    // Open the raw file
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for reading." << std::endl;
        return;
    }

    // Read the uint16_t raw data into dtype pointer
    uint16_t* data = new uint16_t[size];
    file.read(reinterpret_cast<char*>(data), size*sizeof(uint16_t));
    file.close();

    // Convert uint16_t data into dtype data
    for (int i = 0; i < size; ++i) {
        image[i] = static_cast<dtype>(data[i]);
    }

    if(flag_verbose) {
        std::cout << "Data has been successfully read." << std::endl;
    }

    // Clean up
    delete[] data;  
}
template void read_input<int>(int*, const std::string&, const int, const int);
template void read_input<uint16_t>(uint16_t*, const std::string&, const int, const int);
template void read_input<float>(float*, const std::string&, const int, const int);

// Print 2D matrix
template<typename dtype>
void show_matrix_3D(dtype *image, const int xsize, const int ysize, const int zsize) {
    dtype *im = image;
    std::cout << "\nMatrix: (" << xsize << "." << ysize << "." << zsize << ")\n";
    for (int idz = 0; idz < zsize; idz++) {
        std::cout << "\nslice: " << idz << "\n";
        for (int idy = 0; idy < ysize; idy++) {
            for (int idx = 0; idx < xsize; idx++) {
                std::cout << " " << im[idx];
            }
            im += xsize;
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}
template void show_matrix_3D<int>(int *, const int, const int, const int);
template void show_matrix_3D<float>(float *, const int, const int, const int);


//check the results obtained
template<typename dtype>
void check_result(dtype *test, dtype *ref, const int nx, const int ny, const int nz) {

    double epsilon = 1.0E-8;
    bool match = 1;
    dtype *itest = test;
    dtype *iref = ref;
    
    for(int iz=0; iz<nz;iz++){
        for(int iy=0; iy<ny;iy++){
            for (int ix=0; ix<nx; ix++){
                if(std::abs(itest[ix] - iref[ix]) > (dtype)epsilon){
                    match = 0;
                    std::cout << "Matrices do not match!\n";
                    std::cout << "test: " << std::fixed << std::setprecision(2) << std::setw(5) << itest[ix] 
                              << " ref " << std::setw(5) << iref[ix] 
                              << " at curent ["<< ix << "," << iy << "," << iz << "]\n";
                    // break;
                    return;
                }
            }
            itest += nx; iref += nx;
        }
    }
    if(match) std::cout << "Matrices match!\n";
}
template  void check_result<int>(int*,  int*, const int, const int, const int);
template  void check_result<float>(float*,  float*, const int, const int, const int);
