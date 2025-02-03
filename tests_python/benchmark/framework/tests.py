import pandas as pd
from .time import time_module_only
from .time_check import time_compare_and_plot

def run(csv_file, image, operation, machine, ngpus, gpu_flag, repetitions, gpuMemory, kernel):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    print("")
    print(image.shape)
    operation_name = operation[0]
    skimage_func = operation[1]
    harpia_func = operation[2]

    # Call the timing and comparison function
    print(operation_name)
    if(skimage_func):
        time_compare_and_plot(
            csv_data=csv_data,
            hardware=ngpus,
            machine=machine,
            gpuMemory=gpuMemory,
            module_func=harpia_func,
            skimage_func=skimage_func,
            image=image,
            kernel=kernel,
            plot=False,
            show=False,
            operation=operation_name,
            framework="scikit",
            slice_num=0,
            figsize=(18, 6),
            save_path=f"plots/{operation_name.replace(' ', '_')}_{str(image.dtype)}.png",
            repetitions=repetitions,
            gpu=gpu_flag
        )
    else:
        time_module_only(
            csv_data=csv_data,
            hardware=ngpus,
            machine=machine,
            gpuMemory=gpuMemory,
            module_func=harpia_func,
            image=image,
            kernel=kernel,
            plot=False,
            show=False,
            operation=operation_name,
            slice_num=0,
            figsize=(18, 6),
            save_path=f"plots/{operation_name.replace(' ', '_')}_{str(image.dtype)}.png",
            repetitions=repetitions,
            gpu=gpu_flag
        )
    print('\nFinish Test!')
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv(csv_file, index=False)
    return results_df

def run_no_kernel(csv_file, image, operation, machine, ngpus, gpu_flag, repetitions, gpuMemory):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    print("")
    print(image.shape)
    operation_name = operation[0]
    skimage_func = operation[1]
    harpia_func = operation[2]

    # Call the timing and comparison function
    print(operation_name)
    if(skimage_func):
        time_compare_and_plot(
            csv_data=csv_data,
            hardware=ngpus,
            machine=machine,
            gpuMemory=gpuMemory,
            module_func=harpia_func,
            skimage_func=skimage_func,
            image=image,
            plot=False,
            show=False,
            operation=operation_name,
            framework="scikit",
            slice_num=0,
            figsize=(18, 6),
            save_path=f"plots/{operation_name.replace(' ', '_')}_{str(image.dtype)}.png",
            repetitions=repetitions,
            gpu=gpu_flag
        )
    else:
        time_module_only(
            csv_data=csv_data,
            hardware=ngpus,
            machine=machine,
            gpuMemory=gpuMemory,
            module_func=harpia_func,
            image=image,
            plot=False,
            show=False,
            operation=operation_name,
            slice_num=0,
            figsize=(18, 6),
            save_path=f"plots/{operation_name.replace(' ', '_')}_{str(image.dtype)}.png",
            repetitions=repetitions,
            gpu=gpu_flag
        )
    print('\nFinish Tests!')
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv(csv_file, index=False)
    return results_df