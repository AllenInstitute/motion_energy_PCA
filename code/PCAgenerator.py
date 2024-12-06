import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils
from sklearn.decomposition import PCA

class PCAgenerator:
    def __init__(self, motion_zarr_path: str, crop_region = None):
        self.motion_zarr_path = motion_zarr_path
        self.crop = True
        self.loaded_metadata = None
        self.crop_region = crop_region

    def define_crop_region(self, crop_region = None):
        if crop_region is None:
            crop_region=(100, 100, 300, 200)
        # Unpack crop dimensions
        x, y, width, height = crop_region
        # Apply crop using slicing: [y:y+height, x:x+width]
        print(f"Crop region: x={x}, y={y}, width={width}, height={height}")
        self.crop_region=crop_region
        return self

 
    def apply_pca_to_motion_energy(self, n_components=3):
        """Apply PCA to the motion energy."""
        me_store = zarr.DirectoryStore(self.motion_zarr_path)
        frames_me = da.from_zarr(me_store , component='data')
        print(f'Loaded frames {frames_me.shape}')
        if self.crop:
            if self.crop_region is None:
                self.define_crop_region()
            print(f'Applying crop to me frames {self.crop_region}')
            crop_y_start, crop_x_start, crop_y_end, crop_x_end = self.crop_region
            frames_me2 = frames_me[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            H = crop_y_end - crop_y_start
            W = crop_x_end - crop_x_start
            frames_me2 = frames_me2.rechunk((100, H, W)) 
        else:
            frames_me2 = frames_me

        # Reshape frames for PCA: (n_frames, height * width)
        n_frames, height, width = frames_me2.shape
        flattened_frames = frames_me2.reshape(n_frames, height * width)
        
        # Convert Dask array to NumPy array for PCA (PCA requires NumPy array)
        print("Converting frames to NumPy array for PCA...")
        flattened_frames = flattened_frames.compute()
        pca = PCA(n_components=n_components)
        # Apply PCS
        pca_motion_energy = pca.fit_transform(flattened_frames)
        self.pca = pca
        self.pca_motion_energy = pca_motion_energy

        # create spatial masks to see what PCs look like
        spatial_masks = self.compute_spatial_masks(pcs = pca_motion_energy, frames_me2=frames_me2, n_components=n_components, standardize=True)
        self.spatial_masks = spatial_masks
        return pca, pca_motion_energy

       
    def compute_spatial_masks(self, pcs, frames_me2, n_components= None, standardize=True):
        """
        Compute spatial masks from principal components and motion energy.

        Parameters:
        ----------
        pcs : np.ndarray
            A 2D array of shape (n_samples, n_components) containing principal components.
        motion_energy : np.ndarray
            A 3D array of shape (n_samples, height, width) representing motion energy data.
        n : int, optional
            The number of principal components to use. If None, defaults to pcs.shape[1].
        standardize : bool, optional
            Whether to standardize the masks. Defaults to False.

        Returns:
        -------
        spatial_masks : list of np.ndarray
            A list of 2D arrays where each array represents the mean spatial mask for each principal component.
        """
        
        # Set default number of components if n is None
        if n_components is None:
            n_components = pcs.shape[1]

        spatial_masks = []

        for i in range(n_components):
            # Extract the i-th principal component
            pc = pcs[:, i]
            
            # Compute the mask for the current principal component
            masks = frames_me2 * pc[:, np.newaxis, np.newaxis]
            
            # Standardize the mask if required
            if standardize:
                masks_mean = np.mean(masks)
                masks_std = np.std(masks)
                final_masks = (masks - masks_mean) / masks_std
                mean_final_mask = np.mean(final_masks, axis=0)
            else:
                mean_final_mask = np.mean(masks, axis=0)
            
            # Append the computed mask to the list
            spatial_masks.append(mean_final_mask)
        
        return np.array(spatial_masks)


