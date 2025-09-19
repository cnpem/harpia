import GPUtil
from multiprocessing import Process, Event, Queue
import time

class MonitorProcess:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.stop_event = Event()
        self.queue = Queue()

        # Initial memory usage BEFORE subprocess starts
        gpus = GPUtil.getGPUs()
        self.initial_mem = [gpu.memoryUsed for gpu in gpus]

        # Start subprocess
        self.process = Process(target=self._run, args=(self.stop_event, self.queue, delay))
        self.process.start()

    def _run(self, stop_event, queue, delay):
        memory_usage_log = []

        while not stop_event.is_set():
            gpus = GPUtil.getGPUs()
            mem_usage = [gpu.memoryUsed for gpu in gpus]
            memory_usage_log.append(mem_usage)
            time.sleep(delay)

        queue.put(memory_usage_log)

    def stop(self, absolute_max=False):
        self.stop_event.set()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        # Final memory usage (AFTER subprocess ends)
        gpus = GPUtil.getGPUs()
        final_mem = [gpu.memoryUsed for gpu in gpus]

        memory_usage_log = []
        while not self.queue.empty():
            memory_usage_log = self.queue.get()

        if not memory_usage_log:
            return []

        max_mem = [max(mem[i] for mem in memory_usage_log) for i in range(len(memory_usage_log[0]))]

        if absolute_max:
            return max_mem

        return [x for i, m, f in zip(self.initial_mem, max_mem, final_mem) for x in (i, m, f)]
 
 
 ####################################################
from harpia.morphology import morph_2D_chan_vese as cv_harpia
from harpia.morphology import morph_2D_geodesic_active_contour as gac_harpia

from cucim.skimage.segmentation import morphological_chan_vese as cv_cucim
from cucim.skimage.segmentation import morphological_geodesic_active_contour as gac_cucim

from skimage.segmentation import morphological_chan_vese as cv_skimage
from skimage.segmentation import morphological_geodesic_active_contour as gac_skimage
from skimage.segmentation import (
    inverse_gaussian_gradient,
)
import cupy as cp
import numpy as np

def initlevelset(imgshape = (2052,2052),size = 6, function="skimage"):
    init_ls = np.zeros(imgshape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    if function == "skimage":
        return init_ls
    elif function == "cucim":
        return cp.asarray(init_ls)
    elif function == "harpia":
        return init_ls > 0

def image_active_contour(function="skimage"):
    path = '/ibira/lnls/labs/tepui/home/egon.borges/work/dev_annotat3D/harpia_for_tests/harpia/example_images/slice_of_Recon_2052x2052_32bits.raw'
    img = np.fromfile(path, dtype=np.float32).reshape((2052, 2052))
    if function == "cucim":
        return cp.asarray(img)
    else:
        return img

def inverse_gradient_image(function="skimage"):
    img = image_active_contour()
    gimage = inverse_gaussian_gradient(img, sigma=1.0)
    if function == "cucim":
        return cp.asarray(gimage)
    else:
        return gimage
#####################################################################################################################################################
#####################################################################################################################################################
operations_activecontours = [
    {
        "name": "Morphlogical Chan Vese",
        "skimage": cv_skimage,
        "skimage_param": {"image": image_active_contour(), "init_level_set": initlevelset(function="skimage"), 'smoothing': 3},
        "harpia": cv_harpia,
        "harpia_param": {"hostImage": image_active_contour(), "initLs": initlevelset(function="harpia"), 'smoothing': 3, "lambda1": 1.0, "lambda2": 1.0},
        "cucim": cv_cucim,
        "cucim_param": {"image": image_active_contour(function="skimage"), "init_level_set": initlevelset(function="skimage"), 'smoothing': 3},

    },
    {
        "name": "Morphlogical Geodesic Active Contour",
        "skimage": gac_skimage,
        "skimage_param": {"gimage": inverse_gradient_image(), "init_level_set": initlevelset(function="skimage"), 'smoothing': 3, 'balloon': -1, 'threshold': 0.7},
        "harpia": gac_harpia,
        "harpia_param": {"hostImage": inverse_gradient_image(function="harpia"), "initLs": initlevelset(function="harpia"), 'smoothing': 3, 'balloonForce': -1, 'threshold': 0.7},
        "cucim": gac_cucim,
        "cucim_param": {"gimage": inverse_gradient_image(function="skimage"), "init_level_set": initlevelset(function="skimage"), 'smoothing': 3, 'balloon': -1, 'threshold': 0.7},
    },
]
#####################################################################################################################################################
#####################################################################################################################################################
import os
import time
import inspect
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

def time_active_contours_with_iterations(
    csv_data, machine, operation_config, max_iterations=100, save_iterations=True,
    output_dir="active_contour_results", repetitions=1, gpuMemory=0.4, ngpus=-1, show=True
):
    """
    Execute active contour algorithms for all three libraries and save results per iteration.
    
    Parameters:
    -----------
    csv_data : list
        List to store timing and performance data
    machine : str
        Machine identifier for logging
    operation_config : dict
        Configuration dictionary from operations_activecontours
    max_iterations : int
        Maximum number of iterations to run
    save_iterations : bool
        Whether to save intermediate iteration results
    output_dir : str
        Directory to save iteration results
    repetitions : int
        Number of repetitions for timing
    gpuMemory : float
        GPU memory fraction to use
    ngpus : int
        Number of GPUs to use
    show : bool
        Whether to print timing information
    """
    
    operation_name = operation_config["name"]
    
    # Create output directories
    if save_iterations:
        os.makedirs(f"{output_dir}/{operation_name.replace(' ', '_')}", exist_ok=True)
        for lib in ['harpia', 'skimage', 'cucim']:
            if lib in operation_config:
                os.makedirs(f"{output_dir}/{operation_name.replace(' ', '_')}/{lib}", exist_ok=True)
    
    # Storage for timing results
    timing_results = {
        'harpia': {'times': [], 'mem_usage': {}, 'iteration_times': []},
        'skimage': {'times': [], 'mem_usage': {}, 'iteration_times': []},
        'cucim': {'times': [], 'mem_usage': {}, 'iteration_times': []}
    }
    
    # Storage for iteration results
    iteration_results = {
        'harpia': [],
        'skimage': [],
        'cucim': []
    }
    
    print(f"Starting {operation_name} comparison...")
    
    # ---- Execute for each library ----
    for lib in ['harpia', 'skimage', 'cucim']:
        if lib not in operation_config:
            continue
            
        func = operation_config[lib]
        params = operation_config[f"{lib}_param"].copy()
        
        print(f"Running {lib}...")
        
        try:
            if lib == 'harpia':
                result, times, used_gpuMemory, mem_usage, iter_times, iter_results = run_harpia_iterations(
                    func, repetitions, params, max_iterations, save_iterations, 
                    operation_name, output_dir, lib, gpuMemory, ngpus
                )
                
            elif lib == 'skimage':
                result, times, used_gpuMemory, mem_usage, iter_times, iter_results = run_skimage_iterations(
                    func, repetitions, params, max_iterations, save_iterations,
                    operation_name, output_dir, lib
                )
                
            elif lib == 'cucim':
                result, times, mem_usage, iter_times, iter_results = run_cucim_iterations(
                    func, repetitions, params, max_iterations, save_iterations,
                    operation_name, output_dir, lib
                )
                used_gpuMemory = True
                
            timing_results[lib]['times'] = times
            timing_results[lib]['mem_usage'] = mem_usage
            timing_results[lib]['iteration_times'] = iter_times
            iteration_results[lib] = iter_results
            
            print(f"{lib} completed successfully")
            
        except Exception as e:
            print(f"Error running {lib}: {str(e)}")
            timing_results[lib]['times'] = []
            timing_results[lib]['mem_usage'] = {}
            timing_results[lib]['iteration_times'] = []
    
    # ---- Process and log results ----
    log_active_contour_results(
        csv_data, machine, operation_name, timing_results, 
        repetitions, gpuMemory, ngpus, show, operation_config, max_iterations
    )
    
    return timing_results, iteration_results

def run_harpia_iterations(func, repetitions, params, max_iterations, save_iterations, 
                         operation_name, output_dir, lib, gpuMemory, ngpus):
    """Run harpia active contour iteration by iteration."""
    
    # Filter parameters for the function signature
    sig = inspect.signature(func)
    base_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    # Add GPU parameters if available
    if 'gpuMemory' in sig.parameters:
        base_params['gpuMemory'] = gpuMemory
    if 'ngpus' in sig.parameters:
        base_params['ngpus'] = ngpus
    
    all_times = []
    all_iteration_times = []
    all_iteration_results = []
    mem_usage = {}
    
    for rep in range(repetitions):
        rep_times = []
        rep_results = []
        
        # Get initial level set
        current_ls = base_params.get('initLs', base_params.get('init_level_set')).copy()
        
        total_start = time.perf_counter()
        
        if rep == 0:  # Monitor memory only on first repetition
            monitor = MonitorProcess(0.1)
        
        for iteration in range(1, max_iterations + 1):
            # Set parameters for single iteration
            iter_params = base_params.copy()
            iter_params['iterations'] = 1  # Run only 1 iteration
            
            # Update the level set parameter
            if 'initLs' in iter_params:
                iter_params['initLs'] = current_ls
            elif 'init_level_set' in iter_params:
                iter_params['init_level_set'] = current_ls
            
            iter_start = time.perf_counter()
            current_ls = func(**iter_params)
            iter_time = time.perf_counter() - iter_start
            
            rep_times.append(iter_time)
            
            # Save iteration result
            if save_iterations and rep == 0:  # Save only for first repetition
                save_iteration_result(current_ls, iteration, operation_name, output_dir, lib)
            
            if rep == 0:
                rep_results.append({
                    'iteration': iteration,
                    'level_set': current_ls.copy() if hasattr(current_ls, 'copy') else np.array(current_ls)
                })
        
        total_time = time.perf_counter() - total_start
        all_times.append(total_time)
        
        if rep == 0:
            mem_usage = monitor.stop() if 'monitor' in locals() else {}
            all_iteration_times = rep_times
            all_iteration_results = rep_results
    
    used_gpuMemory = 'gpuMemory' in base_params
    return current_ls, all_times, used_gpuMemory, mem_usage, all_iteration_times, all_iteration_results

def run_skimage_iterations(func, repetitions, params, max_iterations, save_iterations,
                          operation_name, output_dir, lib):
    """Run scikit-image active contour iteration by iteration."""
    
    # Filter parameters for the function signature
    sig = inspect.signature(func)
    base_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    all_times = []
    all_iteration_times = []
    all_iteration_results = []
    mem_usage = {}
    
    for rep in range(repetitions):
        rep_times = []
        rep_results = []
        
        # Get initial level set
        current_ls = base_params.get('init_level_set').copy()
        
        total_start = time.perf_counter()
        
        if rep == 0:  # Monitor memory only on first repetition
            monitor = MonitorProcess(0.1)
        
        for iteration in range(1, max_iterations + 1):
            # Set parameters for single iteration
            iter_params = base_params.copy()
            iter_params['num_iter'] = 1  # Run only 1 iteration
            iter_params['init_level_set'] = current_ls
            
            iter_start = time.perf_counter()
            current_ls = func(**iter_params)
            iter_time = time.perf_counter() - iter_start
            
            rep_times.append(iter_time)
            
            # Save iteration result
            if save_iterations and rep == 0:  # Save only for first repetition
                save_iteration_result(current_ls, iteration, operation_name, output_dir, lib)
            
            if rep == 0:
                rep_results.append({
                    'iteration': iteration,
                    'level_set': current_ls.copy()
                })
        
        total_time = time.perf_counter() - total_start
        all_times.append(total_time)
        
        if rep == 0:
            mem_usage = monitor.stop() if 'monitor' in locals() else {}
            all_iteration_times = rep_times
            all_iteration_results = rep_results
    
    return current_ls, all_times, False, mem_usage, all_iteration_times, all_iteration_results

def run_cucim_iterations(func, repetitions, params, max_iterations, save_iterations,
                        operation_name, output_dir, lib):
    """Run CuCIM active contour iteration by iteration with proper timing."""
    
    # Setup memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    def _clear_cupy_memblocks():
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    # Filter parameters for the function signature
    sig = inspect.signature(func)
    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    # Extract CPU arrays that need to be converted
    cpu_image = None
    cpu_init_ls = None
    cpu_gimage = None
    
    # Find the image parameter (could be 'image' or 'gimage')
    if 'image' in filtered_params:
        cpu_image = filtered_params['image']
        if hasattr(cpu_image, 'get'):  # Already a cupy array
            cpu_image = cpu_image.get()
        del filtered_params['image']  # Remove from params, we'll add GPU version
    
    if 'gimage' in filtered_params:
        cpu_gimage = filtered_params['gimage']
        if hasattr(cpu_gimage, 'get'):  # Already a cupy array
            cpu_gimage = cpu_gimage.get()
        del filtered_params['gimage']  # Remove from params, we'll add GPU version
    
    # Find the initial level set parameter
    if 'initLs' in filtered_params:
        cpu_init_ls = filtered_params['initLs']
        if hasattr(cpu_init_ls, 'get'):  # Already a cupy array
            cpu_init_ls = cpu_init_ls.get()
        del filtered_params['initLs']  # Remove from params, we'll add GPU version
    elif 'init_level_set' in filtered_params:
        cpu_init_ls = filtered_params['init_level_set']
        if hasattr(cpu_init_ls, 'get'):  # Already a cupy array
            cpu_init_ls = cpu_init_ls.get()
        del filtered_params['init_level_set']  # Remove from params, we'll add GPU version
    
    all_times = {'total': [], 'memory': [], 'gpu': []}
    all_iteration_times = []
    all_iteration_results = []
    mem_usage = {}
    
    # Warm-up run with memory monitoring
    monitor = MonitorProcess(0.1)
    
    # Convert to GPU for warm-up
    gpu_image = cp.asarray(cpu_image) if cpu_image is not None else None
    gpu_gimage = cp.asarray(cpu_gimage) if cpu_gimage is not None else None
    gpu_init_ls = cp.asarray(cpu_init_ls) if cpu_init_ls is not None else None
    
    # Set GPU parameters for warm-up
    warmup_params = filtered_params.copy()
    if gpu_image is not None:
        warmup_params['image'] = gpu_image
    if gpu_gimage is not None:
        warmup_params['gimage'] = gpu_gimage
    if gpu_init_ls is not None:
        if 'initLs' in sig.parameters:
            warmup_params['initLs'] = gpu_init_ls
        else:
            warmup_params['init_level_set'] = gpu_init_ls
    
    warmup_params['num_iter'] = 1
    output = func(**warmup_params)
    mem_usage = monitor.stop()
    
    # Clean up warm-up
    if gpu_image is not None:
        del gpu_image
    if gpu_gimage is not None:
        del gpu_gimage
    if gpu_init_ls is not None:
        del gpu_init_ls
    _clear_cupy_memblocks()
    
    for rep in range(repetitions):
        rep_times = []
        rep_mem_times = []
        rep_gpu_times = []
        rep_results = []
        
        # Get initial level set for this repetition
        current_ls_cpu = cpu_init_ls.copy()
        
        total_start = time.perf_counter()
        
        for iteration in range(1, max_iterations + 1):
            # === MEMORY TRANSFER TIME ===
            mem_start = time.perf_counter()
            
            # Convert arrays to GPU
            gpu_image = cp.asarray(cpu_image) if cpu_image is not None else None
            gpu_gimage = cp.asarray(cpu_gimage) if cpu_gimage is not None else None
            gpu_current_ls = cp.asarray(current_ls_cpu)
            
            # Prepare parameters
            iter_params = filtered_params.copy()
            iter_params['num_iter'] = 1  # Single iteration
            
            if gpu_image is not None:
                iter_params['image'] = gpu_image
            if gpu_gimage is not None:
                iter_params['gimage'] = gpu_gimage
            if 'initLs' in sig.parameters:
                iter_params['initLs'] = gpu_current_ls
            else:
                iter_params['init_level_set'] = gpu_current_ls
            
            mem_transfer_time = time.perf_counter() - mem_start
            
            # === GPU COMPUTATION TIME ===
            gpu_start = time.perf_counter()
            gpu_result = func(**iter_params)
            gpu_comp_time = time.perf_counter() - gpu_start
            
            # === MEMORY RETRIEVAL TIME ===
            mem_retrieve_start = time.perf_counter()
            current_ls_cpu = gpu_result.get() if hasattr(gpu_result, 'get') else np.array(gpu_result)
            
            # Clean up GPU memory
            if gpu_image is not None:
                del gpu_image
            if gpu_gimage is not None:
                del gpu_gimage
            del gpu_current_ls, gpu_result
            _clear_cupy_memblocks()
            
            mem_retrieve_time = time.perf_counter() - mem_retrieve_start
            
            # Calculate times
            total_mem_time = mem_transfer_time + mem_retrieve_time
            total_iter_time = total_mem_time + gpu_comp_time
            
            rep_times.append(total_iter_time)
            rep_mem_times.append(total_mem_time)
            rep_gpu_times.append(gpu_comp_time)
            
            # Save iteration result (only for first repetition)
            if save_iterations and rep == 0:
                save_iteration_result(current_ls_cpu, iteration, operation_name, output_dir, lib)
            
            if rep == 0:
                rep_results.append({
                    'iteration': iteration,
                    'level_set': current_ls_cpu.copy()
                })
        
        total_time = time.perf_counter() - total_start
        all_times['total'].append(total_time)
        all_times['memory'].append(sum(rep_mem_times))
        all_times['gpu'].append(sum(rep_gpu_times))
        
        if rep == 0:
            all_iteration_times = rep_times
            all_iteration_results = rep_results
    
    return current_ls_cpu, all_times, mem_usage, all_iteration_times, all_iteration_results

def save_iteration_result(level_set, iteration, operation_name, output_dir, lib):
    """Save iteration result to file."""
    
    # Convert to CPU array if needed
    if hasattr(level_set, 'get'):
        level_set_cpu = level_set.get()
    else:
        level_set_cpu = np.array(level_set)
    
    # Save as numpy array
    filename = f"{output_dir}/{operation_name.replace(' ', '_')}/{lib}/iteration_{iteration:03d}.npy"
    np.save(filename, level_set_cpu)
    
    # Save as image every 10 iterations or at specific milestones
    if iteration % 10 == 0 or iteration in [1, 5]:
        plt.figure(figsize=(8, 8))
        plt.imshow(level_set_cpu, cmap='gray')
        plt.title(f"{lib} - {operation_name} - Iteration {iteration}")
        plt.axis('off')
        img_filename = f"{output_dir}/{operation_name.replace(' ', '_')}/{lib}/iteration_{iteration:03d}.png"
        plt.savefig(img_filename, bbox_inches='tight', dpi=150)
        plt.close()

def log_active_contour_results(csv_data, machine, operation_name, timing_results, 
                             repetitions, gpuMemory, ngpus, show, operation_config, max_iterations):
    """Log the timing results to CSV data - ONE ROW PER ITERATION."""
    
    # Get image metadata once
    image = None
    for lib in ['harpia', 'skimage', 'cucim']:
        if f"{lib}_param" in operation_config:
            params = operation_config[f"{lib}_param"]
            for key, value in params.items():
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    image = value
                    break
            if image is not None:
                break
    
    if image is not None:
        if hasattr(image, 'get'):  # CuPy array
            image = image.get()
        image_dtype = str(image.dtype)
        image_size_mb = round(image.nbytes / (1024 ** 2), 1)
        image_shape = image.shape
    else:
        image_dtype = "Unknown"
        image_size_mb = "Unknown"
        image_shape = "Unknown"
    
    # Prepare memory usage data once
    mem_data = {}
    for lib in ['harpia', 'skimage', 'cucim']:
        if timing_results[lib]['mem_usage']:
            if isinstance(timing_results[lib]['mem_usage'], dict):
                for i, mem in enumerate(timing_results[lib]['mem_usage']):
                    mem_data[f"{lib}_gpu{i}(MiB)"] = mem
            elif isinstance(timing_results[lib]['mem_usage'], list):
                for i, mem in enumerate(timing_results[lib]['mem_usage']):
                    mem_data[f"{lib}_gpu{i}(MiB)"] = mem
            else:
                mem_data[f"{lib}_mem(MiB)"] = timing_results[lib]['mem_usage']
    
    # CREATE ONE CSV ROW FOR EACH ITERATION
    for iteration in range(1, max_iterations + 1):
        # Get times for this specific iteration
        harpia_time = "N/A"
        if timing_results['harpia']['iteration_times'] and len(timing_results['harpia']['iteration_times']) >= iteration:
            harpia_time = timing_results['harpia']['iteration_times'][iteration - 1]
        
        skimage_time = "N/A"
        if timing_results['skimage']['iteration_times'] and len(timing_results['skimage']['iteration_times']) >= iteration:
            skimage_time = timing_results['skimage']['iteration_times'][iteration - 1]
        
        cucim_total_time = "N/A"
        cucim_mem_time = "N/A"
        cucim_gpu_time = "N/A"
        
        if timing_results['cucim']['iteration_times'] and len(timing_results['cucim']['iteration_times']) >= iteration:
            cucim_total_time = timing_results['cucim']['iteration_times'][iteration - 1]
            
            # For CuCIM, we need to extract per-iteration memory and GPU times
            # Since we stored them as total times, we need to calculate per iteration
            if timing_results['cucim']['times'] and isinstance(timing_results['cucim']['times'], dict):
                if (timing_results['cucim']['times']['memory'] and 
                    timing_results['cucim']['times']['gpu'] and
                    len(timing_results['cucim']['times']['memory']) > 0):
                    
                    # Get the first repetition's total times and divide by iterations
                    total_mem = timing_results['cucim']['times']['memory'][0]  # First repetition
                    total_gpu = timing_results['cucim']['times']['gpu'][0]     # First repetition
                    
                    # Average per iteration (approximation)
                    cucim_mem_time = total_mem / max_iterations
                    cucim_gpu_time = total_gpu / max_iterations
        
        # Calculate speed ratios for this iteration
        faster_skimage = "N/A"
        faster_cucim = "N/A"
        if harpia_time != "N/A":
            if skimage_time != "N/A":
                faster_skimage = round(skimage_time / harpia_time, 2)
            if cucim_total_time != "N/A":
                faster_cucim = round(cucim_total_time / harpia_time, 2)
        
        # Add ONE ROW for this iteration
        csv_data.append({
            'Operation': operation_name,
            'Machine': machine,
            'Iteration': iteration,  # ITERATION NUMBER HERE!
            'Gpus': ngpus,
            'gpuMemory': gpuMemory,
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
            **mem_data
        })
    
    # Optional print summary
    if show:
        print(f"\n=== {operation_name} Results ===")
        print(f"Total Iterations: {max_iterations}")
        print(f"Logged {max_iterations} rows to CSV (one per iteration)")
        
        # Show first and last iteration times as examples
        if timing_results['harpia']['iteration_times']:
            print(f"Harpia - Iteration 1: {timing_results['harpia']['iteration_times'][0]:.4f}s")
            print(f"Harpia - Iteration {max_iterations}: {timing_results['harpia']['iteration_times'][-1]:.4f}s")
        
        if timing_results['skimage']['iteration_times']:
            print(f"Scikit - Iteration 1: {timing_results['skimage']['iteration_times'][0]:.4f}s")
            print(f"Scikit - Iteration {max_iterations}: {timing_results['skimage']['iteration_times'][-1]:.4f}s")
        
        if timing_results['cucim']['iteration_times']:
            print(f"Cucim - Iteration 1: {timing_results['cucim']['iteration_times'][0]:.4f}s")
            print(f"Cucim - Iteration {max_iterations}: {timing_results['cucim']['iteration_times'][-1]:.4f}s")

# Usage example:
def run_active_contour_comparison(max_iterations=10, repetitions=3):
    """Run active contour comparison with specified parameters."""
    
    csv_data = []
    machine = "test_machine"
    
    for operation_config in operations_activecontours:
        timing_results, iteration_results = time_active_contours_with_iterations(
            csv_data=csv_data,
            machine=machine,
            operation_config=operation_config,
            max_iterations=max_iterations,
            save_iterations=False,
            output_dir="active_contour_comparison",
            repetitions=repetitions,
            gpuMemory=0.4,
            ngpus=1,
            show=True
        )
        
        print(f"Completed {operation_config['name']}")
        print("-" * 50)
    
    return csv_data, timing_results, iteration_results

#####################################################################################################################################################
#####################################################################################################################################################

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Active Contour Benchmarking Tool')
    parser.add_argument('--max_iterations', '-i', type=int, default=10,
                        help='Maximum number of iterations to run (default: 10)')
    parser.add_argument('--repetitions', '-r', type=int, default=3,
                        help='Number of repetitions for timing (default: 3)')
    parser.add_argument('--output_file', '-o', type=str, default='morphological_active_contour_files.csv',
                        help='Output CSV file name (default: morphological_active_contour_files.csv)')
    parser.add_argument('--machine', '-m', type=str, default='test_machine',
                        help='Machine identifier for logging (default: test_machine)')
    parser.add_argument('--no_save_iterations', action='store_true',
                        help='Disable saving intermediate iteration results')
    parser.add_argument('--gpu_memory', type=float, default=0.4,
                        help='GPU memory fraction to use (default: 0.4)')
    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ACTIVE CONTOUR BENCHMARKING")
    print("="*60)
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Output File: {args.output_file}")
    print(f"Machine: {args.machine}")
    print(f"Save Iterations: {not args.no_save_iterations}")
    print(f"GPU Memory: {args.gpu_memory}")
    print(f"Number of GPUs: {args.ngpus}")
    print("="*60)
    
    # Run the benchmark
    csv_data = []
    
    for operation_config in operations_activecontours:
        timing_results, iteration_results = time_active_contours_with_iterations(
            csv_data=csv_data,
            machine=args.machine,
            operation_config=operation_config,
            max_iterations=args.max_iterations,
            save_iterations=not args.no_save_iterations,
            output_dir="active_contour_comparison",
            repetitions=args.repetitions,
            gpuMemory=args.gpu_memory,
            ngpus=args.ngpus,
            show=True
        )
        
        print(f"Completed {operation_config['name']}")
        print("-" * 50)
    
    # Save results to CSV
    print('\nFinish Test!')
    results_df = pd.DataFrame(csv_data)
    # Append to the file, only writing the header if the file does not exist
    results_df.to_csv(args.output_file, mode='a', header=not os.path.exists(args.output_file), index=False)
    print(f'Saved Tests to {args.output_file}!')
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total CSV rows saved: {len(csv_data)}")
    print(f"Operations tested: {len(operations_activecontours)}")
    print(f"Iterations per operation: {args.max_iterations}")
    print(f"Repetitions: {args.repetitions}")
