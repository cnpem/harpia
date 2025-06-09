import timeit                           # For timing the function
import numpy as np                     # For array manipulation
import matplotlib.pyplot as plt         # For plotting images
import cupy as cp
import cucim


# Merged function to time, compare, and plot results with MSE instead of accuracy
def time_compare(csv_data, machine, module_func, skimage_func, cucim_func, image, 
                          kernel=None, show=False, operation="", repetitions=1, gpuMemory =0.4, 
                          ngpus = -1, *args, **kwargs):
    module_times = []
    skimage_times = []
    cucim_times = []

    # Time the module function multiple times
    if(kernel is None):
        module_output = module_func(image, *args, gpuMemory= gpuMemory, **kwargs) #warm up run
        for _ in range(repetitions):
            start = timeit.default_timer()
            module_output = module_func(image, *args, gpuMemory= gpuMemory, ngpus = ngpus, **kwargs)
            module_times.append(timeit.default_timer() - start)

        # Time the scikit-image function multiple times
        skimage_output = skimage_func(image) #warm up run
        for _ in range(repetitions):
            start = timeit.default_timer()
            skimage_output = skimage_func(image)
            skimage_times.append(timeit.default_timer() - start)
    
        # Time the cucim function multiple times
        image_cucim = cp.asarray(image)
        cucim_output = cucim_func(image_cucim) #warm up run
        for _ in range(repetitions):
            start = timeit.default_timer()
            cucim_output = cucim_func(image)
            cucim_times.append(timeit.default_timer() - start)
        cucim_output = cucim_output.get()

    else:
        module_output = module_func(image, kernel, *args, gpuMemory= gpuMemory, **kwargs) #warm up run
        for _ in range(repetitions):
            start = timeit.default_timer()
            module_output = module_func(image, kernel, *args, gpuMemory= gpuMemory, ngpus = ngpus, **kwargs)
            module_times.append(timeit.default_timer() - start)

        # Time the scikit-image function multiple times
        skimage_output = skimage_func(image, kernel) #warm up run
        for _ in range(repetitions):
            start = timeit.default_timer()
            skimage_output = skimage_func(image, kernel)
            skimage_times.append(timeit.default_timer() - start)
        
        # Time the cucim function multiple times
        image_cucim = cp.asarray(image)
        kernel_cucim = cp.asarray(kernel)
        cucim_output = cucim_func(image_cucim, kernel_cucim) #warm up run
        for _ in range(repetitions):
            start = timeit.default_timer()
            cucim_output = cucim_func(image_cucim, kernel_cucim)
            cucim_times.append(timeit.default_timer() - start)
        cucim_output = cucim_output.get()

    # Calculate mean times
    module_time = np.mean(module_times)
    skimage_time = np.mean(skimage_times)
    cucim_time = np.mean(cucim_times)

    # Calculate Time ratio
    faster_skimage = round(skimage_time/module_time, 2)
    faster_cucim = round(module_time/cucim_time, 2)

    # Get the image data type, size, and dimensions
    image_dtype = str(image.dtype)
    image_size_mb = round(image.nbytes/(1024 ** 2),1)
    image_shape = (image.shape[2], image.shape[1], image.shape[0]) #compatible with c++ shape

    # Add timing, MSE, and image details to CSV data
    csv_data.append({
        'Operation':  module_func.__name__ if not operation else operation,
        'Machine': machine,
        'Gpus': ngpus,
        'gpuMemory': gpuMemory, 
        'Module Time (s)': module_time,
        'Scikit-Image Time (s)': skimage_time,
        'Scikit Time Ratio': faster_skimage,
        'Cucim Time (s)': cucim_time,
        'Cucim Time Ratio': faster_cucim,
        'Repetitions' : repetitions,
        'Image Data Type': image_dtype,
        'Image Size (MB)': image_size_mb,
        'Image Dimensions': image_shape
    })

    # Print statistics
    if show:
        print(f"Operation: {module_func.__name__ if not operation else operation}")
        print(f"Module Time: {module_time:.4f} seconds")
        print(f"Image Data Type: {image_dtype}")
        print(f"Image Size: {image_size_mb} MB")
        print(f"Image Dimensions: {image_shape}")