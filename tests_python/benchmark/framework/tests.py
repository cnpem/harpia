import os
import pandas as pd
from .time import time_module_only
from .time_check import time_compare

def run(csv_file, image, operation, machine, ngpus, repetitions, gpuMemory, kernel):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    print("")
    print(image.shape)
    operation_name = operation[0]
    skimage_func = operation[1]
    harpia_func = operation[2]
    cucim_func = operation[3]

    # Call the timing and comparison function
    time_compare(
        csv_data=csv_data,
        machine=machine,
        gpuMemory=gpuMemory,
        ngpus=ngpus,
        module_func=harpia_func,
        skimage_func=skimage_func,
        cucim_func=cucim_func,
        image=image,
        kernel=kernel,
        operation=operation_name,
        repetitions=repetitions
    )

    print('\nFinish Test!')
    results_df = pd.DataFrame(csv_data)
    # Append to the file, only writing the header if the file does not exist
    results_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    return results_df

def run_no_kernel(csv_file, image, operation, machine, ngpus, repetitions, gpuMemory):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    print("")
    print(image.shape)
    operation_name = operation[0]
    skimage_func = operation[1]
    harpia_func = operation[2]
    cucim_func = operation[3]

    # Call the timing and comparison function
    print(operation_name)
    if(skimage_func):
        time_compare(
            csv_data=csv_data,
            machine=machine,
            gpuMemory=gpuMemory,
            ngpus=ngpus,
            module_func=harpia_func,
            skimage_func=skimage_func,
            cucim_func=cucim_func,
            image=image,
            operation=operation_name,
            repetitions=repetitions
        )
    else:
        time_module_only(
            csv_data=csv_data,
            machine=machine,
            gpuMemory=gpuMemory,
            ngpus=ngpus,
            module_func=harpia_func,
            image=image,
            operation=operation_name,
            repetitions=repetitions
        )
    print('\nFinish Tests!')
    results_df = pd.DataFrame(csv_data)
    # Append to the file, only writing the header if the file does not exist
    results_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    print('\nSaved Tests!')

    return results_df