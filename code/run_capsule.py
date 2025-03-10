import os
from tqdm import tqdm
import utils
import time  # Added for timing
from PCAgenerator import PCAgenerator
from pathlib import Path
import numpy as np

DATA_PATH = utils.get_data_path(pipeline=True)
#DATA_PATH = Path("/data/")
zarr_paths = utils.find_zarr_file(DATA_PATH)
#zarr_paths = utils.find_input_paths(directory = DATA_PATH, return_file=False, endswith='zarr')
print(len(zarr_paths))
def run():
    #for zarr_path, npz_path in zip(zarr_paths[:1], npz_paths[:1]):
    for zarr_path in zarr_paths:
        print(f'...Loading {zarr_path}')
        try:
            pkl_file = utils.find_input_paths(directory = zarr_path.parent, return_file = True, endswith='.pkl')[0]
        except:
            pkl_file = None
        try:
            npz_file = utils.find_input_paths(directory = zarr_path.parent, return_file = True, endswith='.npz')[0]
        except:
            npz_file = None

        start_time = time.time()  # Start the timer

        me_pca = PCAgenerator(motion_zarr_path = zarr_path, pkl_file = pkl_file, npz_file=npz_file, use_cropped_frames=True, standardize4PCA=False) 
        
        me_pca, post_crop_frames_me = me_pca._apply_pca_to_motion_energy_without_dask()

        me_pca._add_spatial_masks(me_pca.pca_motion_energy, post_crop_frames_me)

        me_pca._save_results()
        
    
        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()