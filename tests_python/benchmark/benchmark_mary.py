#!/usr/bin/env python3
import os
import time
import argparse
from tqdm import tqdm
from datetime import datetime

from framework import image, tests
from framework import operations

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Run benchmark tests")
parser.add_argument("--img_num", type=int, default=4, help="Image number (1 or 4)")
parser.add_argument("--framework", type=str, default=None, help="Filter by specific framework")
args = parser.parse_args()

# --- Image Configuration ---
image_configs = {
    1: {
        "name": "small",
        "xsize": 190, "ysize": 207, "zsize": 100,
        "grayscale_path": "../../example_images/grayscale/crua_A_190x207x100_16b.raw",
        "binary_path": "../../example_images/binary/crua_A_190x207x100_16b.raw",
        "grayscale_dtype": "uint16",
        "binary_dtype": "uint16"
    },
    4: {
        "name": "big",
        "xsize": 2052, "ysize": 2052, "zsize": 2048,
        "grayscale_path": "../../example_images/grayscale/Recon_2052x2052x2048_32bits.raw",
        "binary_path": "../../example_images/binary/Recon_2052x2052x2048_16bits.raw",
        "grayscale_dtype": "float32",
        "binary_dtype": "uint16"
    }
}

# --- Settings ---
machine = 'mary'
ngpus = 1
gpuMemory = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
repetitions = 1
reps = 30

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/{timestamp}"
os.makedirs(results_dir, exist_ok=True)
log_file_path = os.path.join(results_dir, "log.txt")
log_file = open(log_file_path, "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

# --- Load Selected Image ---
cfg = image_configs[args.img_num]
log(f"Reading {cfg['name']} image...")
start = time.time()
image_grayscale = image.load(cfg['grayscale_path'], cfg['xsize'], cfg['ysize'], cfg['zsize'], cfg['grayscale_dtype'])
image_binary = image.load(cfg['binary_path'], cfg['xsize'], cfg['ysize'], cfg['zsize'], cfg['binary_dtype'])
end = time.time()
log("Finished reading image!")
log(f"Reading time: {end - start:.2f}s")

# --- Image dtypes ---
image_types_grayscale = [
    "float32",
    # "int32",
    # "uint32",
]

image_types_binary = [
    "int32",
    # "int16",
    # "uint16",
    # "uint32",
]

# --- Configurations ---
configurations = [
    {
        "framework": "harpia",
        "image_type": "binary",
        "nslices": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "dtypes": image_types_binary,
        "image": image_binary,
        "operations": operations.harpia_binary
    },
    # {
    #     "framework": "harpia",
    #     "image_type": "grayscale",
    #     "nslices": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    #     "dtypes": image_types_grayscale,
    #     "image": image_grayscale,
    #     "operations": operations.harpia_grayscale
    # },
    # {
    #     "framework": "harpia",
    #     "image_type": "grayscale",
    #     "nslices": [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    #     "dtypes": image_types_grayscale,
    #     "image": image_grayscale,
    #     "operations": operations.harpia_gauss #these operations give segemntation fault for 2 slices
    # },
    {
        "framework": "skimage",
        "image_type": "binary",
        "nslices": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], #bigger iamges take too long to execute
        "dtypes": image_types_binary,
        "image": image_binary,
        "operations": operations.skimage_binary
    },
    {
        "framework": "skimage",
        "image_type": "grayscale",
        "nslices": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], #bigger iamges take too long to execute
        "dtypes": image_types_grayscale,
        "image": image_grayscale,
        "operations": operations.skimage_grayscale
    },
    {
        "framework": "cucim",
        "image_type": "binary",
        "nslices": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], #maximun size executable in one DGX GPU
        "dtypes": image_types_binary,
        "image": image_binary,
        "operations": operations.cucim_bianry
    },
    {
        "framework": "cucim",
        "image_type": "grayscale",
        "nslices": [1, 2, 4, 8, 16, 32, 64, 128],  #maximun size executable in one DGX GPU
        "dtypes": image_types_grayscale,
        "image": image_grayscale,
        "operations": operations.cucim_grayscale
    },
    {
        "framework": "cucim",
        "image_type": "grayscale",
        "nslices": [256, 512],  #only these operations execute for these sizes in one DGX GPU
        "dtypes": image_types_grayscale,
        "image": image_grayscale,
        "operations": operations.cucim_grayscale_no_threashold
    }
]

# --- Run Tests ---

log(f"\nMachine '{machine}' with {reps} reps, gpuMemory {gpuMemory},image #{args.img_num} ({cfg['name']}).")

start = time.time()
for gpuMem in gpuMemory:
    for config in configurations:
        if args.framework and config["framework"] != args.framework:
            continue

        for nslices in config["nslices"]:
            csv_file = os.path.join(results_dir, f"{machine}_{reps}reps_{config['framework']}_{nslices}_{config['image_type']}_{gpuMem}gpuMem.csv")
            log(f"\nSaving to file: {csv_file}")
            for dtype in config["dtypes"]:
                image_input = config["image"].astype(dtype=dtype)
                image_input = image_input[:nslices, :, :]
                for _ in tqdm(range(reps), desc=f"{config['framework']}_{config['image_type']}_{dtype}_{nslices}"):
                    for operation in config["operations"]:
                        kernel = operation["kernel"]
                        try:
                            tests.run(
                                csv_file,
                                image_input,
                                operation,
                                machine,
                                ngpus,
                                repetitions,
                                gpuMem,
                                kernel
                            )
                        except Exception as e:
                            log(f"[ERROR] Framework: {config['framework']} | Op: {kernel} | Slices: {nslices} | Dtype: {dtype} | gpuMem: {gpuMem}\n  {str(e)}")

end = time.time()
log("\n✅ All tests completed!")
log(f"Total executin time: {end - start:.2f}s")

log_file.close()
