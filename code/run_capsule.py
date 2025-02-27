import os
from tqdm import tqdm
import utils
import time  # Added for timing
from PCAgenerator import PCAgenerator
from pathlib import Path

DATA_PATH = Path("/root/capsule/data/")
zarr_paths = utils.find_files(root_dir = DATA_PATH, endswith='zarr')
npz_paths = utils.find_files(root_dir = DATA_PATH, endswith='.npz', return_dir=False)
print(len(zarr_paths))
assert len(zarr_paths) == len(npz_paths), 'zarr files and npz files are misaligned'

def run():
    for zarr_path, npz_path in zip(zarr_paths, npz_paths):
        start_time = time.time()  # Start the timer

        me_pca = PCAgenerator(zarr_path, npz_path, use_cropped_frames=True, standardize4PCA=False) 
        
        me_pca, post_crop_frames_me = me_pca._apply_pca_to_motion_energy_without_dask()

        me_pca._add_spatial_masks(me_pca.pca_motion_energy, post_crop_frames_me)

        me_pca._save_results()
        
    
        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()