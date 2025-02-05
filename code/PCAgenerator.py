import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from tqdm import tqdm

class PCAgenerator:
    def __init__(self, motion_zarr_path: str, crop_region = None,
    standardize4PCA = True):
        self.motion_zarr_path = motion_zarr_path
        self.crop = True
        self.loaded_metadata = None
        self.crop_region = crop_region
        self.n_components = 100 # to fit PCA
        self.n_to_plot = 3 # to show
        self.standardize4PCA = standardize4PCA
        self.chunk_size = 100

    def _define_crop_region(self, crop_region = None):
        #check metadata:
        try:
            crop_region = self.loaded_metadata['crop_region']
    
        if crop_region is None:
            crop_region=(100, 100, 200, 300)
        # Unpack crop dimensions
        y, x, height, width = crop_region
        # Apply crop using slicing: [y:y+height, x:x+width]
        print(f"Crop region: x={x}, y={y}, width={width}, height={height}")
        self.crop_region=crop_region
        return self

    def _apply_pca_to_motion_energy_without_dask(self):
        """Apply PCA to the motion energy."""
        # Open the Zarr store and load the 'data' array
        me_store = zarr.DirectoryStore(self.motion_zarr_path)
        zarr_group = zarr.open(me_store, mode='r')
        frames_me = zarr_group['data']
        print(f'Loaded frames {frames_me.shape}')

        # Determine the number of components
        n_components = self.n_components if self.n_components is not None else 100

        # Crop frames if needed
        if self.crop:
            if self.crop_region is None:
                self._define_crop_region()
            print(f'Applying crop to ME frames {self.crop_region}')
            crop_y_start, crop_x_start, crop_y_end, crop_x_end = self.crop_region
            frames_me = frames_me[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            H = crop_y_end - crop_y_start
            W = crop_x_end - crop_x_start
        else:
            H, W = frames_me.shape[1:]

        # Initialize Incremental PCA
        ipca = IncrementalPCA(n_components=n_components)

        # Standardize if required
        if self.standardize4PCA:
            print("Standardizing data...")
            mean = np.zeros(H * W)
            std = np.zeros(H * W)

        # Process data chunk by chunk
        print("Fitting PCA in chunks...")
        start_index = 20000
        for i in tqdm(range(start_index, frames_me.shape[0], self.chunk_size)):
            chunk = frames_me[i:i + self.chunk_size]

            # check to make sure the last chunk is not too short 
            # ideally chunk.shape[0] should be >= n_components
            # which means that chunk_size should be >= n_components
            next_chunk = frames_me[i + self.chunk_size:]
            print(next_chunk.shape)
            if next_chunk.shape[0] < n_components:
                chunk = frames_me[i:] # get the rest of frames 
                print(f'processing last chunk, shape: {chunk.shape}')

            chunk_flattened = chunk.reshape(chunk.shape[0], -1)

            # if self.standardize4PCA:
            #     if i == 0:  # Compute mean and std on the first chunk
            #         mean = chunk_flattened.mean(axis=0)
            #         std = chunk_flattened.std(axis=0)
            #     chunk_flattened = (chunk_flattened - mean) / std
            
            ipca.partial_fit(chunk_flattened)

        print("PCA fitting complete.")

        # Transform data in chunks
        print("Transforming data in chunks...")
        transformed_chunks = []
        for i in range(start_index, frames_me.shape[0], self.chunk_size):
            chunk = frames_me[i:i + self.chunk_size]
            chunk_flattened = chunk.reshape(chunk.shape[0], -1)

            if self.standardize4PCA:
                chunk_flattened = (chunk_flattened - mean) / std

            transformed_chunk = ipca.transform(chunk_flattened)
            transformed_chunks.append(transformed_chunk)

        # Combine transformed chunks into a single array
        pca_motion_energy = np.vstack(transformed_chunks)
        self.pca = ipca
        self.pca_motion_energy = pca_motion_energy
        print('Added PCA results.')

        # Create spatial masks to visualize PCs
        print('Computing spatial masks...')
        spatial_masks = self._compute_spatial_masks_in_chunks(pca_motion_energy, frames_me, standardize=True)
        print('Done.')
        self.spatial_masks = spatial_masks

        return ipca, pca_motion_energy


    def _compute_spatial_masks_in_chunks(self, pcs, frames_me2, standardize=True):
        """
        Compute spatial masks from principal components and motion energy.

        Parameters:
        ----------
        pcs : np.ndarray
            A 2D array of shape (n_samples, n_components) containing principal components.
        frames_me2 : zarr.Array
            A Zarr array of shape (n_samples, height, width) representing motion energy data.
        standardize : bool, optional
            Whether to standardize the masks. Defaults to True.

        Returns:
        -------
        spatial_masks : list of np.ndarray
            A list of 2D arrays where each array represents the mean spatial mask for each principal component.
        """

        # Set default number of components if not specified
        n_components = self.n_to_plot if self.n_to_plot is not None else 3
        spatial_masks = []

        # Iterate over principal components
        for i in range(n_components):
            # Extract the i-th principal component
            pc = pcs[:, i]
            mask_sum = None
            count = 0

            # Iterate over chunks of frames_me2
            for chunk_start in range(0, frames_me2.shape[0], self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, frames_me2.shape[0])

                # Load the chunk from Zarr
                chunk = frames_me2[chunk_start:chunk_end]

                # Apply the principal component to the chunk
                chunk_masks = chunk * pc[chunk_start:chunk_end, np.newaxis, np.newaxis]

                # Accumulate the sum of masks
                if mask_sum is None:
                    mask_sum = np.sum(chunk_masks, axis=0)
                else:
                    mask_sum += np.sum(chunk_masks, axis=0)

                # Update the count
                count += chunk_masks.shape[0]

            # Compute the mean mask
            mean_mask = mask_sum / count

            # Standardize the mask if required
            if standardize:
                mean = mean_mask.mean()
                std = mean_mask.std()
                mean_mask = (mean_mask - mean) / std

            # Append the computed mean mask to the list
            spatial_masks.append(mean_mask)

        return np.array(spatial_masks)

    def _plot_spatial_masks(self):
            
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

    def _plot_explained_variance(self):
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

    def _plot_pca_components_traces(self, component_indices=[0, 1, 2], axes=None):
        """
        Plots 3 PCA components from pca_motion_energy against x_trace_seconds.

        - component_indices: list of indices for the PCA components to plot (default: [0, 1, 2])
        - title: title of the plot
        """
        if axes is None:
            fig, axes = plt.subplots(len(component_indices), 1,figsize=(15,2*len(component_indices)))
        pca_motion_energy = self.pca_motion_energy
        fps = self.loaded_metadata['fps']
        title_fontsize = 20
        label_fontsize = 16
        tick_fontsize = 14
        if pca_motion_energy.shape[1] < 3:
            raise ValueError("pca_motion_energy must have at least 3 components to plot.")
        
        x_range = 10000
        
        x_trace_seconds = np.round(np.arange(1, x_range) / fps, 2)
        for i, ax in enumerate(axes):
            ax.plot(x_trace_seconds, pca_motion_energy[np.arange(1, x_range), component_indices[i]])
            ax.set_ylabel(f'PCA {component_indices[i] + 1}', fontsize = label_fontsize)
            ax.set_title(f'PCA {component_indices[i] + 1} over time (s)', fontsize = title_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.grid(True)
        
        axes[-1].set_xlabel('Time (s)', fontsize = label_fontsize)
        
        plt.tight_layout()
        return axes


## extra
 
    def _apply_pca_to_motion_energy(self):
        """Apply PCA to the motion energy."""
        me_store = zarr.DirectoryStore(self.motion_zarr_path)
        frames_me = da.from_zarr(me_store , component='data')
        print(f'Loaded frames {frames_me.shape}')

        # Get number of components to fit
        if self.n_components is None:
            n_components = 100
        else:
            n_components = self.n_components

        # crop frames if needed
        if self.crop:
            if self.crop_region is None:
                self._define_crop_region()
            print(f'Applying crop to me frames {self.crop_region}')
            crop_y_start, crop_x_start, crop_y_end, crop_x_end = self.crop_region
            frames_me2 = frames_me[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end].copy()
            del frames_me
            H = crop_y_end - crop_y_start
            W = crop_x_end - crop_x_start
            # frames_me2 = frames_me2.rechunk((self.chunk_size, H, W)) 
        else:
            frames_me2 = frames_me
        

        # Reshape frames for PCA: (n_frames, height * width)
        n_frames, height, width = frames_me2.shape
        new_shape = (n_frames, height * width)
        new_chunks = (self.chunk_size, new_shape[0])
        try: 
            reshaped_frames = frames_me2.reshape(new_shape).rechunk(new_chunks)
            print(f"new reshaped array array shape: {reshaped_frames.shape}")
            print(f"rechunked array size: {reshaped_frames.chunksize}")
        except ValueError as e:
            print(f"Error during reshaping: {e}")

        # Standardize pixels if required
        if self.standardize4PCA:
            print("Standardizing data...")
            mean = reshaped_frames.mean(axis=0)
            std = reshaped_frames.std(axis=0)
            standardized_frames = (reshaped_frames - mean) / std
            print('done.')
        else:
            print("Skipping standardization of data.")
            standardized_frames = reshaped_frames

        # plot random frame
        index = np.random(frames_me2.shape[0])
        plt.imshow(frames_me2[index],vmax=np.percentile(frames_me2[index].ravel(), 98))
        plt.show()


        # Apply PCA to chunks
        ipca = IncrementalPCA(n_components=n_components)

        # Incrementally fit PCA on Dask array
        print("Fitting PCA in chunks...")
        for chunk in standardized_frames.to_delayed():
            chunk_data = chunk[0].compute()
            ipca.partial_fit(chunk_data)
        print('done.')
        # Transform data in chunks
        print("transforming data in chunks...")
        transformed_chunks = [
             da.from_array(ipca.transform(chunk[0].compute()), chunks=(self.chunk_size, n_components))
            for chunk in standardized_frames.to_delayed()]
        print('done.')
        # Combine transformed chunks into a single array
        pca_motion_energy = da.concatenate(transformed_chunks, axis=0)
        self.pca = ipca
        self.pca_mpotion_energy = pca_motion_energy
        print('added PCA results.')
        # try using regular PCA
        # pca = PCA(n_components=n_components)
        # pca_motion_energy = pca.fit_transform(standardized_frames)
        # self.pca = pca
        # self.pca_motion_energy = pca_motion_energy
        
        # create spatial masks to see what PCs look like
        print('Computing spatial masks...')
        spatial_masks = self._compute_spatial_masks(pcs = pca_motion_energy, frames_me2=frames_me2, standardize=True)
        print('done.')
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
