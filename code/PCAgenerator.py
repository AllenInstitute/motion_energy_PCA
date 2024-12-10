import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAgenerator:
    def __init__(self, motion_zarr_path: str, pkl_file : str, crop_region = None):
        self.motion_zarr_path = motion_zarr_path
        self.pkl_file = pkl_file
        self.crop = True
        self.loaded_metadata = None
        self.crop_region = crop_region
        self.n_components = 100 # to fit PCA
        self.n_to_plot = 3 # to show

    def _define_crop_region(self, crop_region = None):
        #check metadata:
        crop_region = utils.check_crop_region(self.pkl_file)
        if crop_region is None:
            crop_region=(100, 100, 300, 200)
        # Unpack crop dimensions
        x, y, width, height = crop_region
        # Apply crop using slicing: [y:y+height, x:x+width]
        print(f"Crop region: x={x}, y={y}, width={width}, height={height}")
        self.crop_region=crop_region
        return self

 
    def _apply_pca_to_motion_energy(self, n_components=None):
        """Apply PCA to the motion energy."""
        me_store = zarr.DirectoryStore(self.motion_zarr_path)
        frames_me = da.from_zarr(me_store , component='data')
        print(f'Loaded frames {frames_me.shape}')
        if self.n_components is None:
            n_components = 3
        else:
            n_components = self.n_components

        if self.crop:
            if self.crop_region is None:
                self._define_crop_region()
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
        spatial_masks = self._compute_spatial_masks(pcs = pca_motion_energy, frames_me2=frames_me2, standardize=True)
        self.spatial_masks = spatial_masks
        return pca, pca_motion_energy

       
    def _compute_spatial_masks(self, pcs, frames_me2, standardize=True):
        """
        Compute spatial masks from principal components and motion energy.

        Parameters:
        ----------
        pcs : np.ndarray
            A 2D array of shape (n_samples, n_components) containing principal components.
        motion_energy : np.ndarray
            A 3D array of shape (n_samples, height, width) representing motion energy data.
        
        standardize : bool, optional
            Whether to standardize the masks. Defaults to False.

        Returns:
        -------
        spatial_masks : list of np.ndarray
            A list of 2D arrays where each array represents the mean spatial mask for each principal component.
        """
        
        # Set default number of components if n is None
        n_components = self.n_to_plot
        if n_components is None:
            n_components = 3

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

    def plot_spatial_masks(self):
            
        n_components = self.n_components
            
        fig = plt.figure(figsize=(3 * n_components, 2))
        
        for i, mask in enumerate(self.spatial_masks):
            plt.subplot(1, n_components, i + 1)
            plt.imshow(mask, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(label='')
            plt.axis('off')
            plt.title(f'PC {i + 1} mask')
        
        plt.show()
        
        return fig

    def plot_explained_variance(self):
        """
        Plots the explained variance ratio of each principal component from a PCA model.

        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot of explained variance.
        """
        fig = plt.figure(figsize=(4,2))
        fontsize=12
        # Check if pca has been fitted
        pca = self.pca
        if not hasattr(pca, 'explained_variance_ratio_'):
            raise ValueError("PCA object must be fitted before plotting.")
        
        # Get the explained variance ratio and convert to percentage
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = explained_variance_ratio * 100
        
        # Plot the explained variance
        
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance, 'o-', linewidth=2, markersize=5)
        plt.title('Variance Explained by Principal Components', fontsize=fontsize)
        plt.xlabel('Principal Component', fontsize=fontsize)
        plt.ylabel('Explained Variance (%)', fontsize=fontsize)
        plt.xlim([0, 30]) #show only top 30
        plt.tight_layout()
        #plt.xticks(range(1, len(explained_variance_ratio) + 1))
        #plt.grid(True)
        plt.show()

        return fig

    def plot_pca_components_traces(self, component_indices=[0, 1, 2], axes=None):
        """
        Plots 3 PCA components from pca_motion_energy against x_trace_seconds.

        - component_indices: list of indices for the PCA components to plot (default: [0, 1, 2])
        - title: title of the plot
        """
        pca_motion_energy = self.pca_motion_energy
        fps = utils.get_fps(self.pkl_file)
        title_fontsize = 20
        label_fontsize = 16
        tick_fontsize = 14
        if pca_motion_energy.shape[1] < 3:
            raise ValueError("pca_motion_energy must have at least 3 components to plot.")
        
        for i, ax in enumerate(axes):
            ax.plot(x_trace_seconds, pca_motion_energy[:, component_indices[i]])
            ax.set_ylabel(f'PCA {component_indices[i] + 1}', fontsize = label_fontsize)
            ax.set_title(f'PCA {component_indices[i] + 1} over time (s)', fontsize = title_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.grid(True)
        
        axes[-1].set_xlabel('Time (s)', fontsize = label_fontsize)
        
        
        return axes

