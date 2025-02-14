import os
from tqdm import tqdm
import utils
import time  # Added for timing
from PCAgenerator import PCAgenerator

zarr_paths = utils.find_files(root_dir = '/root/capsule/data', endswith='zarr')
npz_paths = utils.find_files(root_dir = '/root/capsule/data', endswith='.npz', return_dir=False)
crop_region = (200, 290, 280, 360)
assert len(zarr_paths) == len(npz_paths), 'zarr files and npz files are misaligned'

def run():
    for zarr_path, npz_path in zip(zarr_paths, npz_paths):
        start_time = time.time()  # Start the timer

        me_pca = PCAgenerator(zarr_path, npz_path, crop=True, crop_region=crop_region, standardize4PCA=False, standardizeMasks=True) 
        
        me_pca, post_crop_frames_me = me_pca._apply_pca_to_motion_energy_without_dask()
        
        me_pca._save_results()

        me_pca._add_spatial_masks(me_pca.pca_motion_energy, post_crop_frames_me)

        me_pca._save_results()
        
        #plot and save fig
        fig = me_pca._plot_spatial_masks()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_spatial_masks.npg', dpi=300, bbox_inches="tight", transparent=False)

        fig = me_pca._plot_pca_components_traces()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_components_traces.npg', dpi=300, bbox_inches="tight", transparent=False)

        fig = me_pca._plot_explained_variance()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_explained_variance.npg', dpi=300, bbox_inches="tight", transparent=False)

        fig = me_pca._plot_motion_energy_trace()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'motion_energy_trace.npg', dpi=300, bbox_inches="tight", transparent=False)

    
        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()