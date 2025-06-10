import timeit  
import inspect                         # For timing the function
import numpy as np                     # For array manipulation
import matplotlib.pyplot as plt         # For plotting images
import cupy as cp
import cucim

def time_compare(
    csv_data, machine, module_func, skimage_func, skimage_param, cucim_func, image, 
    kernel=None, show=True, operation="", repetitions=1, gpuMemory=0.4, ngpus=-1, 
    *args, **kwargs):

    module_times, skimage_times, cucim_times = [], [], []
    module_time = skimage_time = cucim_time = "N/A"
    faster_skimage = faster_cucim = "N/A"

    def time_function(func, *f_args, **f_kwargs):
        sig = inspect.signature(func)
        parameters = sig.parameters

        filtered_kwargs = {}
        used_gpuMemory = False

        for k, v in f_kwargs.items():
            if k in parameters:
                filtered_kwargs[k] = v
                if k == 'gpuMemory':
                    used_gpuMemory = True

        output = func(*f_args, **filtered_kwargs)  # Warm-up run
        times = []
        for _ in range(repetitions):
            start = timeit.default_timer()
            output = func(*f_args, **filtered_kwargs)
            times.append(timeit.default_timer() - start)
        return output, times, used_gpuMemory

    # ---- Module (Harpia) ----
    if module_func:
        if kernel is not None:
            module_output, module_times, used_gpuMemory = time_function(
                module_func, image, kernel, *args, gpuMemory=gpuMemory, ngpus=ngpus, **kwargs
            )
        else:
            module_output, module_times, used_gpuMemory = time_function(
                module_func, image, *args, gpuMemory=gpuMemory, ngpus=ngpus, **kwargs
            )
        print("harpia finished")

    # ---- Skimage ----
    if skimage_func:
        if kernel is not None:
            skimage_output, skimage_times, used_gpuMemory = time_function(skimage_func, image, kernel)
        elif skimage_param:
            skimage_output, skimage_times, used_gpuMemory = time_function(skimage_func, image, **skimage_param)
        else:
            skimage_output, skimage_times, used_gpuMemory = time_function(skimage_func, image)
        print("skimage finished")

    # ---- CuCIM ----
    if cucim_func:
        image_cucim = cp.asarray(image)
        if kernel is not None:
            kernel_cucim = cp.asarray(kernel)
            cucim_output, cucim_times, used_gpuMemory = time_function(cucim_func, image_cucim, kernel_cucim)
        else:
            cucim_output, cucim_times, used_gpuMemory = time_function(cucim_func, image_cucim)
        cucim_output = cucim_output.get()
        print("cucim finished")

    # ---- Post Processing ----
    if module_times: module_time = np.mean(module_times)
    if skimage_times:
        skimage_time = np.mean(skimage_times)
        if module_times:
            faster_skimage = round(skimage_time / module_time, 2)
    if cucim_times:
        cucim_time = np.mean(cucim_times)
        if module_times:
            faster_cucim = round(module_time / cucim_time, 2)

    logged_gpuMemory = gpuMemory if used_gpuMemory else 0

    # ---- Image Metadata ----
    image_dtype = str(image.dtype)
    image_size_mb = round(image.nbytes / (1024 ** 2), 1)
    image_shape = (image.shape[0], image.shape[1], image.shape[2])  # (X, Y, Z)

    # ---- CSV Logging ----
    csv_data.append({
        'Operation': operation or module_func.__name__,
        'Machine': machine,
        'Gpus': ngpus,
        'gpuMemory': logged_gpuMemory,
        'Module Time (s)': module_time,
        'Scikit-Image Time (s)': skimage_time,
        'Scikit Time Ratio': faster_skimage,
        'Cucim Time (s)': cucim_time,
        'Cucim Time Ratio': faster_cucim,
        'Repetitions': repetitions,
        'Image Data Type': image_dtype,
        'Image Size (MB)': image_size_mb,
        'Image Dimensions': image_shape
    })

    # ---- Optional Print ----
    if show:
        print(f"Operation: {operation or module_func.__name__}")
        print(f"Module Time: {module_time:.4f} seconds")
        print(f"Scikit Time: {skimage_time:.4f} seconds")
        print(f"Cucim Time: {cucim_time:.4f} seconds")
        print(f"Image Data Type: {image_dtype}")
        print(f"Image Size: {image_size_mb} MB")
        print(f"Image Dimensions: {image_shape}")