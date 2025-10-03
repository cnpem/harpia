# Harpia

# HARPIA

## Description

**Harpia** (High Algorithmic Performance for Image Analysis) is a CUDA-accelerated library for large-scale volumetric image processing and segmentation, developed at the [Brazilian Synchrotron Light Laboratory (LNLS)](https://lnls.cnpem.br/).  
It provides a scalable backend for scientific imaging applications, tightly integrated with [Annotat3D](https://github.com/cnpem/annotat3d), enabling interactive segmentation of datasets that exceed single-GPU memory capacity.

Harpia emphasizes **strict memory control**, **chunked execution**, and a **modular C++/CUDA architecture**, making it well-suited for **HPC and remote multi-user environments**. It delivers performance and scalability beyond widely used frameworks such as cuCIM and scikit-image.

Main features include:

- **Chunked-based Execution**: Native GPU-aware partitioning of large volumes, ensuring scalability and continuity across chunk boundaries.
- **Memory-Safe GPU Resource Management**: Predictable allocation/deallocation to support concurrent, multi-user HPC workflows.
- **High-Performance Filtering Suite**: CUDA implementations of anisotropic diffusion, median, Gaussian, non-local means, unsharp mask, Sobel, and Prewitt filters.
- **Morphological Operations**: Fully GPU-accelerated 2D/3D erosions, dilations, openings, closings, geodesic reconstruction, smoothing, fill holes, and snakes.
- **Thresholding Methods**: CUDA implementations of Otsu, Sauvola, Niblack, adaptive mean, and adaptive Gaussian.
- **Superpixel and Feature Extraction**: GPU kernels for superpixel segmentation, feature pooling, and descriptors such as Local Binary Patterns (LBP) and Hessian-based measures.
- **Quantification Tools**: Area, volume, perimeter, fraction, connected components, and OpenMP-accelerated Euclidean Distance Transform.
- **Annotation Support**: Accelerated active contours, watershed, and intuitive 2D/3D editing modules for human-in-the-loop segmentation.

---

## Project Status

Harpia is under **active development**. Current releases provide stable GPU-accelerated modules and Python bindings. Future work includes:

- Multi-GPU scheduling and heterogeneous computing support
- Dynamic tuning of chunk sizes
- Integration with advanced AI/ML models for segmentation refinement

---

## Repository Structure

The repository is organized into the following main directories:

### **Core Library: `harpia/`**
Python bindings and Cython interfaces to CUDA/C++ kernels.

- **`filters.pyx`**, **`morphology.pyx`**, **`segmentation.pyx`**, **`threshold.pyx`**: Core GPU-accelerated operations.
- **`featureExtraction/`**: Pixel- and superpixel-level feature extraction kernels.
- **`localBinaryPattern/`**, **`shapeIndex/`**: Texture and shape descriptors.
- **`common/`**: Shared utilities (e.g., memory handling, grid/block setup).

### **CUDA Kernels: `src/`**
Low-level CUDA C++ implementations.

- **`filters/`**: Image enhancement and denoising filters.
- **`morphology/`**: Binary and grayscale morphological operators, including geodesic morphological operators. Active contour tools with morphological snakes.
- **`superpixelExtraction/`**: Superpixel and Pixel Feature Extraction.
- **`threshold/`** Local and Global Thresholding Techniques such as Otsu, Niiblack, Gaussian etc.
- **`quantification/`**: Geometric descriptors and connected components.
- **`localBinary/`**: Image enhancement and denoising filters.


### **Headers: `include/`**
C++ header files exposing public APIs for filters, morphology, quantification, etc.

### **Containerization: `container/`**
HPC and reproducibility support.

- **Singularity and Docker recipes** (HPCCM-based) for CUDA, cuDNN, OpenMPI, HDF5, Conda environments.

### **Documentation: `docs/`**
Sphinx + Doxygen documentation sources.

- `docs/src/harpia/*.rst`: API references and tutorials.
- `docs/src/morphology/*.rst`: Detailed operator descriptions.

### **Tests**
- **`tests_cuda/`**: C++/CUDA unit tests.
- **`tests_python/`**: Python notebooks and scripts for correctness and benchmarking.
- **`tests_python/benchmark/`**: Reproducible performance comparisons (scikit-image, cuCIM, Harpia).

## Install

1. Build singularity
```
sudo -s singularity build harpia.sif container/Singularity.def
```

2. Access singularity. Replace '/path/to/harpia' with the directory where you downloaded the Harpia project.
```
singularity shell --nv -B /ibira /path/to/harpia/harpia.sif
bash
```

3. Create new environment
```
conda create -n harpia python=3.9 -y
```

4. Activate it
```
conda activate harpia
```

5. Install requirements
```
pip install -r requirements.txt
```

6. Install harpia
```
python3 setup.py build
pip install dist/harpia-2.3.3-cp39-cp39-linux_x86_64.whl
```

7. Check if installation was succeesfull
```
python3 tests_python/compilation_test.py
```

## Install-dev

1. Build singularity
```
sudo -s singularity build harpia.sif container/Singularity.def
```

2. Access singularity. Replace '/path/to/harpia' with the directory where you downloaded the Harpia project.
```
singularity shell --nv -B /ibira /path/to/harpia/harpia.sif
bash
```

3. Create new environment
```
conda create -n harpia-dev python=3.9 -y
```

4. Activate it
```
conda activate harpia-dev
```

5. Install requirements
```
pip install -r requirements-dev.txt
```

6. Install harpia
```
python3 setup.py build
pip install dist/harpia-2.3.3-cp39-cp39-linux_x86_64.whl
```

7. Check if installation was succeesfull
```
python3 tests_python/compilation_test.py
```

8. Install cucim dependencies:
   1. for cuda 11
   ```
   pip install cupy-cuda11x==13.5.1
   pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cucim-cu11==24.8.0
   ```
   2. for cuda 12
   ```
   pip install cupy-cuda12x==13.6.0
   pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cucim-cu12


### **Need Help?**

If you need help getting started or have any questions, feel free to:
- Open a discussion in the [Discussions](https://github.com/cnpem/harpia/discussions) tab.
- Reach out to the maintainers listed in the [Contributors](#contributors) section.

We look forward to your contributions! 🎉

---

## Contributors

- 👤 **Allan Pinto**
- 👤 **Egon Borges**
- 👤 **Ricardo Grangeiro**
- 👤 **Camila Machado de Araújo**

## References
   ```
- C. M. de Araujo, E. P. B. S. Borges, R. M. C. Grangeiro, A. Pinto.  
  *Advancing Annotat3D with Harpia: A CUDA-Accelerated Library For Large-Scale Volumetric Data Segmentation.*  
  LNLS/CNPEM, 2025.

   ```

