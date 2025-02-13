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
        pca_me._add_spatial_masks(pca_motion_energy, post_crop_frames_me)
        
        #plot and save fig
        fig = me_pca._plot_spatial_masks()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_spatial_masks.npg', dpi=300, bbox_inches="tight", transparent=False)

        fig = me_pca._plot_pca_components_traces()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_components_traces.npg', dpi=300, bbox_inches="tight", transparent=False)

        fig = me_pca._plot_explained_variance()
        utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_explained_variance.npg', dpi=300, bbox_inches="tight", transparent=False)

    
        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()