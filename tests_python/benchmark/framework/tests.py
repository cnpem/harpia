import os
import pandas as pd
from .time import time_compare

def run(csv_file, image, operation, machine, ngpus, repetitions, gpuMemory, kernel):
    # Loop over hardware and GPU parameters, images, and operations
    csv_data = []
    print("")
    print(image.shape)
    operation_name = operation.get('name', None)
    skimage_func = operation.get('skimage', None)
    skimage_param = operation.get('skimage_param', {})
    harpia_func = operation.get('harpia', None)  
    harpia_param = operation.get('harpia_param', {})
    cucim_func = operation.get('cucim', None)
    cucim_param = operation.get('cucim_param', {})

    # Call the timing and comparison function
    if kernel is not None:
        time_compare(
            csv_data=csv_data,
            machine=machine,
            gpuMemory=gpuMemory,
            ngpus=ngpus,
            harpia_func=harpia_func,
            harpia_param=harpia_param,
            skimage_func=skimage_func,
            skimage_param=skimage_param,
            cucim_func=cucim_func,
            cucim_param=cucim_param,
            image=image,
            kernel=kernel,
            operation=operation_name,
            repetitions=repetitions
        )
    else:
        time_compare(
            csv_data=csv_data,
            machine=machine,
            gpuMemory=gpuMemory,
            ngpus=ngpus,
            harpia_func=harpia_func,
            harpia_param=harpia_param,
            skimage_func=skimage_func,
            skimage_param=skimage_param,
            cucim_func=cucim_func,
            cucim_param=cucim_param,
            image=image,
            operation=operation_name,
            repetitions=repetitions
        )

    print('\nFinish Test!')
    results_df = pd.DataFrame(csv_data)
    # Append to the file, only writing the header if the file does not exist
    results_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    print('\nSaved Tests!')

    return results_df
