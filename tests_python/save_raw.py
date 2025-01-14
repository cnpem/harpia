import h5py
import numpy as np

# Define file paths
input_file = '/ibira/lnls/labs/tepui/home/camila.araujo/images/Recon_fdk_recon_raft_cal_39kev_tomo2_cal_pco_2x_39_z1z2_2470_z1_1620_eps_2325_expt_2_5_s_000.hdf5'
output_file = '../example_images/grayscale/Recon_2048x2052x2052_32bits.raw'

# Open the HDF5 file
with h5py.File(input_file, 'r') as hdf:
    # Assuming the dataset is named 'data'
    dataset_name = 'data'  # Replace with the correct dataset name
    data = hdf[dataset_name]
    
    # Extract slices 1500 to 1505
    slices = data[0:]
    
    # Save as .raw file
    slices.tofile(output_file)
    #data.tofile(output_file)

#print(f"Slices 1500 to 1505 saved to {output_file}")
print(f"Data saved to {output_file}")