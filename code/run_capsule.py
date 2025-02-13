import os
from tqdm import tqdm
import utils
import time  # Added for timing
from PCAgenerator import PCAgenerator

zarr_paths = utils.find_zarr_paths()

def run():
    for zarr_path in zarr_paths:
        start_time = time.time()  # Start the timer

        me_pca = PCAgenerator(zarr_files[0], crop = True, crop_region=(250, 300,  400, 500), standardize4PCA=False, standardizeMasks=True) 
        ipca, pca_motion_energy, post_crop_frames_me = me_pca._apply_pca_to_motion_energy_without_dask()

        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()