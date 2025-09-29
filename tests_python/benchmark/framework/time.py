import time  
import inspect                         # For timing the function
import numpy as np                     # For array manipulation
import matplotlib.pyplot as plt         # For plotting images
import cupy as cp
import cucim

from .gpu_memmory import MonitorProcess

def time_function(func, repetitions, *f_args, **f_kwargs):

    sig = inspect.signature(func)
    parameters = sig.parameters

    filtered_kwargs = {}
    used_gpuMemory = False

    for k, v in f_kwargs.items():
        if k in parameters:
            filtered_kwargs[k] = v
            if k == 'gpuMemory':
                used_gpuMemory = True

    # Filter out 'None' from positional arguments (this deals with existing or not 'kernel')
    filtered_args = tuple(arg for arg in f_args if arg is not None)

    # Warm-up run-------
    monitor = MonitorProcess(0.1)
    output = func(*filtered_args, **filtered_kwargs)  
    mem_usage = monitor.stop()
    #-------------------

    # Warm-up run-------
    #mem_usage = ['N/A'] #temp -> use this warm up version only for faster tests
    #--------------------

    # Time several runs
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = func(*filtered_args, **filtered_kwargs)
        times.append(time.perf_counter() - start)

    return output, times, used_gpuMemory, mem_usage



def time_cucim(func, repetitions, image, kernel, **f_kwargs):
    sig = inspect.signature(func)
    parameters = sig.parameters

    filtered_kwargs = {}

    for k, v in f_kwargs.items():
        if k in parameters:
            filtered_kwargs[k] = v

    times = []
    times_memory = []
    times_gpu = []

    # Clear gpu memory after cucim execution
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    def _clear_cupy_memblocks():
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
    if kernel is not None:
        monitor = MonitorProcess(0.1)
        image_cucim = cp.asarray(image)
        kernel_cucim = cp.asarray(kernel)
        output = func(image_cucim, kernel_cucim, **filtered_kwargs)  # Warm-up run
        del image_cucim
        del kernel_cucim
        _clear_cupy_memblocks()
        mem_usage = monitor.stop()
        for _ in range(repetitions):
            start = time.perf_counter()
            image_cucim = cp.asarray(image)
            kernel_cucim = cp.asarray(kernel)
            time_aux = time.perf_counter() - start

            start = time.perf_counter()
            cucim_output = func(image_cucim, kernel_cucim, **filtered_kwargs)
            time_gpu = time.perf_counter() - start

            start = time.perf_counter()
            output = cucim_output.get()
            del image_cucim
            del kernel_cucim
            _clear_cupy_memblocks()
            time_mem = time_aux + time.perf_counter() - start

            times.append(time_mem+time_gpu)
            times_memory.append(time_mem)
            times_gpu.append(time_gpu)

    else:
        monitor = MonitorProcess(0.1)
        image_cucim = cp.asarray(image)
        output = func(image_cucim, **filtered_kwargs)  # Warm-up run
        mem_usage = monitor.stop()
        del image_cucim
        _clear_cupy_memblocks()
        for _ in range(repetitions):
            start = time.perf_counter()
            image_cucim = cp.asarray(image)
            time_aux = time.perf_counter() - start

            start = time.perf_counter()
            cucim_output = func(image_cucim, **filtered_kwargs)
            time_gpu = time.perf_counter() - start

            start = time.perf_counter()
            output = cucim_output.get()
            del image_cucim
            _clear_cupy_memblocks()
            time_mem = time_aux + time.perf_counter() - start

            times.append(time_mem+time_gpu)
            times_memory.append(time_mem)
            times_gpu.append(time_gpu)
            
    return output, {'total':times, 'memory':times_memory, 'gpu':times_gpu}, mem_usage

def time_compare(
    csv_data, machine, harpia_func, skimage_func, cucim_func, image, skimage_param = {},
    harpia_param = {}, cucim_param = {}, kernel=None, show=True, operation="", 
    repetitions=1, gpuMemory=0.4, ngpus=-1):

    print("harpia_func:", harpia_func)

    harpia_times, skimage_times, cucim_times = [], [], []
    harpia_mem, skimage_mem, cucim_mem  = {}, {}, {}
    harpia_time = skimage_time = cucim_total_time = cucim_mem_time = cucim_gpu_time = "N/A"
    faster_skimage = faster_cucim = "N/A"
    used_gpuMemory = False

    # ---- Harpia ----
    if harpia_func:
        _, harpia_times, used_gpuMemory, module_mem_usage = time_function(
            harpia_func, repetitions, image, kernel, gpuMemory=gpuMemory, ngpus=ngpus, 
            **harpia_param
        )
        harpia_mem = {f"harpia_gpu{i}(MiB)": mem for i, mem in enumerate(module_mem_usage)}
        print("harpia finished")

    # ---- Skimage ----
    if skimage_func:
        _, skimage_times, used_gpuMemory, skimage_mem_usage = time_function(
            skimage_func, repetitions, image, kernel, **skimage_param
        )
        skimage_mem = {f"skimage_gpu{i}(MiB)": mem for i, mem in enumerate(skimage_mem_usage)}
        print("skimage finished")

    # ---- CuCIM ----
    if cucim_func:
        _, cucim_times, cucim_mem_usage = time_cucim(
            cucim_func, repetitions, image, kernel, **cucim_param
        )
        cucim_mem = {f"cucim_gpu{i}(MiB)": mem for i, mem in enumerate(cucim_mem_usage)}
        print("cucim finished")

    # ---- Post Processing ----
    if harpia_times: harpia_time = np.mean(harpia_times)
    if skimage_times:
        skimage_time = np.mean(skimage_times)
        if harpia_times:
            faster_skimage = round(skimage_time / harpia_time, 2)
    if cucim_times:
        cucim_total_time = np.mean(cucim_times['total'])
        cucim_mem_time = np.mean(cucim_times['memory'])
        cucim_gpu_time = np.mean(cucim_times['gpu'])
        if harpia_times:
            faster_cucim = round(cucim_total_time / harpia_time, 2)

    logged_gpuMemory = gpuMemory if used_gpuMemory else 0

    # ---- Image Metadata ----
    image_dtype = str(image.dtype)
    image_size_mb = round(image.nbytes / (1024 ** 2), 1) #binary megabyte unit MiB (nvidia-smi compatible)
    image_shape = (image.shape[0], image.shape[1], image.shape[2])  # (X, Y, Z)

    # ---- CSV Logging ----
    csv_data.append({
        'Operation': operation or harpia_func.__name__,
        'Machine': machine,
        'Gpus': ngpus,
        'gpuMemory': logged_gpuMemory,
        'Harpia Time (s)': harpia_time,
        'Scikit Time (s)': skimage_time,
        'Scikit Time Ratio': faster_skimage,
        'Cucim Total Time (s)': cucim_total_time,
        'Cucim Memory Time (s)': cucim_mem_time,
        'Cucim Gpu Time (s)': cucim_gpu_time,
        'Cucim Time Ratio': faster_cucim,
        'Repetitions': repetitions,
        'Image Data Type': image_dtype,
        'Image Size (MiB)': image_size_mb,
        'Image Dimensions': image_shape,
        **harpia_mem,
        **skimage_mem,
        **cucim_mem
    })


    # ---- Optional Print ----
    if show:
        print(f"Operation: {operation or harpia_func.__name__}")
        print(f"Harpia Time: {harpia_time} seconds")
        print(f"Scikit Time: {skimage_time} seconds")
        print(f"Cucim Time: {cucim_total_time} seconds")
        print(f"Image Data Type: {image_dtype}")
        print(f"Image Size: {image_size_mb} MiB")
        print(f"Image Dimensions: {image_shape}")