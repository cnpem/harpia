import h5py
import numpy as np

# Define file paths
input_file = '/ibira/lnls/labs/tepui/home/camila.araujo/images/Recon_fdk_recon_raft_cal_39kev_tomo2_cal_pco_2x_39_z1z2_2470_z1_1620_eps_2325_expt_2_5_s_000.hdf5'
output_file = '../example_images/grayscale/Recon_2052x2052x2048_32bits.raw'

# Open the HDF5 file
with h5py.File(input_file, 'r') as hdf:
    # Assuming the dataset is named 'data'
    dataset_name = 'data'  # Replace with the correct dataset name
    data = hdf[dataset_name][:]
    
    # Check for NaN values and replace with zeros
    data = np.nan_to_num(data, nan=0.0)
    # Binarize: Set all nonzero values to 1
    # data = (data != 0).astype(np.int8)  # This converts to 0s and 1s
    # Save the data as a .raw file
    output_file = '../example_images/grayscale/Recon_2052x2052x2048_8b.raw'
    data.tofile(output_file)

print(f"Data saved to {output_file}")