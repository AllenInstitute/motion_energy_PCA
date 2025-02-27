import os
import json
import pickle
import numpy as np
import zarr
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import logging
import utils
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAgenerator:
    """
    PCA Generator for motion energy data.

    This class applies PCA to motion energy frames stored in a Zarr dataset,
    including cropping, standardization, and chunk-wise processing.
    """

    def __init__(self, motion_zarr_path: str, npz_path: str, recrop: bool = None, crop_region: tuple = None,
                 standardize4PCA: bool = False):
        """
        Initialize PCA Generator.

        Args:
            motion_zarr_path (str): Path to the Zarr dataset containing motion energy frames.
            recrop (bool, optional): Whether to crop frames before PCA. Defaults to None.
            crop_region (tuple, optional): Crop region as (y_start, x_start, y_end, x_end).
            standardize4PCA (bool, optional): If True, standardizes frames before PCA. Defaults to True.
            standardizeMasks (bool, optional): If True, standardizes masks when plotting. Defaults to False.
        """
        self.motion_zarr_path = motion_zarr_path
        self.npz_path = npz_path
        self.recrop = recrop
        self.crop_region = crop_region
        self.use_cropped_frames = False
        self.n_components = 100  # Number of PCA components
        self.n_to_plot = 3  # Number of components to visualize
        self.standardize4PCA = standardize4PCA
        self.chunk_size = 100
        self.start_index = 0  # First frame with data info should have been dropped when me was computed
        self.mean = None
        self.std = None

        self._load_metadata()
        self.me_metadata['crop'] = True # find why it was saved as False
        self._compare_crop_settings()
        self._get_motion_energy_trace()

    
    def _compare_crop_settings(self) -> None:
        """
        Compares the current crop settings with the metadata and determines
        whether cropping should be applied or skipped.
        """
        if self.recrop is True:
            if self.crop_region is None:
                raise ValueError("Crop region cannot be None when crop is set to True")
            
            if self.me_metadata.get("crop") is True:
                if str(self.crop_region) == str(self.me_metadata.get("crop_region")):
                    print("Skipping cropping since frames were already cropped to the right size in Motion Energy Capsule.")
                    self.use_cropped_frames = True
                else:
                    print("Re-cropping frames since the new crop region differs from the one provided in Motion Energy Capsule.")

        elif self.recrop is None:
            if self.me_metadata.get("crop") is True:
                self.use_cropped_frames = True
                print("Crop was not specified, using cropped frames from Motion Energy Capsule.")
            elif self.me_metadata.get("crop") is False:
                print("Computing PCA on full frames.")

        return self

    def _load_metadata(self) -> None:
        """Load metadata from the Zarr store."""
        root_group = zarr.open_group(self.motion_zarr_path, mode='r')
        all_metadata = json.loads(root_group.attrs['metadata'])
        self.video_metadata = all_metadata.get('video_metadata')
        me_metadata = all_metadata.pop('video_metadata',None) #remove video metadata
        self.me_metadata = me_metadata
        logger.info("Metadata loaded successfully.")

        return self

    def _define_crop_region(self, crop_region: tuple = None) -> None:
        """Define and validate the crop region."""
        if crop_region is None:
            crop_region = self.video_metadata.get('crop_region', (100, 100, 200, 300))
            logger.warning(f"Crop region not found in metadata, defaulting to {crop_region}")

        y, x, height, width = crop_region
        logger.info(f"Using crop region: x={x}, y={y}, width={width}, height={height}")
        self.crop_region = crop_region

    def _apply_pca_to_motion_energy_without_dask(self):
        """
        Apply PCA to motion energy frames.

        Returns:
            tuple: (PCA model, transformed PCA motion energy, cropped frames).
        """
        # Load motion energy frames from Zarr
        me_store = zarr.DirectoryStore(self.motion_zarr_path)
        zarr_group = zarr.open(me_store, mode='r')
        if self.use_cropped_frames:
            frames_me = zarr_group['cropped_frames']
        else:
            frames_me = zarr_group['full_frames']
        logger.info(f"Loaded motion energy frames with shape {frames_me.shape}")

        # Define PCA components
        n_components = self.n_components or 100

        # Apply cropping if needed
        if self.recrop and self.use_cropped_frames is False:
            if not self.crop_region:
                raise ValueError("Crop region cannot be None or empty when crop is set to True")
            else:
                crop_y_start, crop_x_start, crop_y_end, crop_x_end = self.crop_region
                frames_me = frames_me[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # Standardize if required (see below)
        if self.standardize4PCA:
            logger.info("Standardizing data before PCA.")
            
        num_frames = frames_me.shape[0]
        logger.info(f"Processing {num_frames - self.start_index} frames, starting at frame {self.start_index}.")

        # Initialize Incremental PCA
        ipca = IncrementalPCA(n_components=n_components)

        # Fit PCA in chunks
        for i in tqdm(range(self.start_index, num_frames, self.chunk_size), desc="Fitting PCA"):
            chunk = frames_me[i:i + self.chunk_size]

            # If the last chunk is too small, include the remaining frames
            if i + self.chunk_size >= num_frames - n_components:
                chunk = frames_me[i:]
                logger.info(f"Processing last chunk with shape {chunk.shape}")

            chunk_flattened = chunk.reshape(chunk.shape[0], -1)

            if self.standardize4PCA:
                chunk_flattened = self._standardize_chunk(chunk_flattened, i)

            ipca.partial_fit(chunk_flattened)

            # Break early if we processed the last chunk
            if i + self.chunk_size >= num_frames - n_components:
                print(f'transform last {i}')
                break

        logger.info("PCA fitting complete.")

        # Transform data in chunks
        transformed_chunks = []
        for i in tqdm(range(self.start_index, num_frames, self.chunk_size), desc="Transforming PCA"):
            chunk = frames_me[i:i + self.chunk_size]

            # Merge last small chunk if needed
            if i + self.chunk_size >= num_frames - n_components:
                chunk = frames_me[i:]

            chunk_flattened = chunk.reshape(chunk.shape[0], -1)

            if self.standardize4PCA:
                chunk_flattened = (chunk_flattened - self.mean) / self.std

                # Check for NaN values after standardization
                nan_count = np.isnan(chunk_flattened).sum()
                if nan_count > 0:
                    logger.warning(f"Standardized data contains {nan_count} NaN values.")

            transformed_chunks.append(ipca.transform(chunk_flattened))

            # Break early if we processed the last chunk
            if i + self.chunk_size >= num_frames - n_components:
                print(f'transform last {i}')
                break

        # Combine transformed chunks into a single array
        pca_motion_energy = np.vstack(transformed_chunks)

        logger.info(f"PCA transformation complete. Output shape: {pca_motion_energy.shape}")
        self.pca = ipca
        self.pca_motion_energy = pca_motion_energy

        return self, frames_me

    def _standardize_chunk(self, chunk_flattened: np.ndarray, i: int) -> np.ndarray:
        """
        Standardizes a given chunk for PCA processing.

        Args:
            chunk_flattened (numpy.ndarray): 2D array where each row represents a sample.
            i (int): Current chunk index.

        Returns:
            numpy.ndarray: Standardized chunk.
        """
        if self.standardize4PCA:
            if i == self.start_index:
                self.mean = chunk_flattened.mean(axis=0)
                self.std = chunk_flattened.std(axis=0)

                if np.any(np.isnan(self.mean)) or np.any(np.isnan(self.std)):
                    raise ValueError("NaN detected in mean or standard deviation!")
                if np.any(self.std == 0):
                    raise ValueError("Standard deviation contains zero values!")

            chunk_flattened = (chunk_flattened - self.mean) / self.std

            # Check for NaN values after standardization
            nan_count = np.isnan(chunk_flattened).sum()
            if nan_count > 0:
                logger.warning(f"Standardized data contains {nan_count} NaN values.")

        return chunk_flattened


    def _add_spatial_masks(self, pca_motion_energy: np.ndarray, post_crop_frames_me: zarr.Array) -> None:
        """
        Computes and stores spatial masks for PCA visualization.

        Args:
            pca_motion_energy (np.ndarray): A 2D array (n_samples, n_components) with PCA-transformed motion energy data.
            post_crop_frames_me (zarr.Array): A 3D array (n_samples, height, width) representing motion energy frames.

        Updates:
            self.spatial_masks (np.ndarray): Stores computed spatial masks for visualization.
        """
        logger.info("Computing spatial masks...")
        self.spatial_masks = self._compute_spatial_masks_in_chunks(pca_motion_energy, post_crop_frames_me)
        logger.info("Spatial masks computation complete.")

        return self


    def _compute_spatial_masks_in_chunks(self, pca_motion_energy: np.ndarray, post_crop_frames_me: zarr.Array) -> np.ndarray:
        """
        Computes spatial masks from principal components and motion energy.

        Args:
            pca_motion_energy (np.ndarray): 2D array (n_samples, n_components) containing PCA components.
            post_crop_frames_me (zarr.Array): 3D array (n_samples, height, width) representing motion energy frames.

        Returns:
            np.ndarray: Array of spatial masks, where each mask represents a mean projection of one principal component.

        Raises:
            ValueError: If array dimensions do not match expectations.
        """

        frames_indx = [100:200]
        # Ensure correct dimensions
        if pca_motion_energy.ndim != 2:
            raise ValueError("pca_motion_energy must be a 2D array (n_samples, n_components).")

        if post_crop_frames_me.ndim != 3:
            raise ValueError("post_crop_frames_me must be a 3D array (n_samples, height, width).")

        # Determine the number of components to process
        n_components = self.n_to_plot if self.n_to_plot is not None else 3
        spatial_masks = []

        logger.info(f"Number of PCA components: {pca_motion_energy.shape[1]}")
        logger.info(f"Total frames available: {post_crop_frames_me.shape[0] - self.start_index}")

        # Standardization flag
        if self.standardizeMasks:
            logger.info("Standardizing PC mask values for plotting.")
        else:
            logger.info("Skipping standardization of PC masks.")

        self.mean_me_frame = np.mean(post_crop_frames_me[frames_indx], axis=0)
        # Iterate over the first n principal components
        for pc_index in range(n_components):
            logger.info(f"Processing Principal Component {pc_index + 1}...")

            # Extract the current principal component
            pc = pca_motion_energy[:, pc_index]

            # Initialize mask sum and count
            mask_sum = np.zeros(post_crop_frames_me.shape[1:], dtype=np.float64)
            count = 0
            num_frames = post_crop_frames_me.shape[0]

            # Process frames in chunks
            for i in tqdm(range(self.start_index, num_frames, self.chunk_size), desc=f"PC {pc_index + 1}"):
                pc_i = i  # Adjust index since PC has one less frame
                chunk = post_crop_frames_me[i:i + self.chunk_size]
                pc_chunk = pc[pc_i:pc_i + self.chunk_size, np.newaxis, np.newaxis]

                # Handle last chunk separately
                if i + self.chunk_size >= num_frames - n_components:
                    chunk = post_crop_frames_me[i:]
                    pc_chunk = pc[pc_i:, np.newaxis, np.newaxis]
                    logger.info(f"Processing last chunk of PC {pc_index + 1}, shape: {chunk.shape}")

                # Ensure chunk and PC sizes match
                if len(chunk) != len(pc_chunk):
                    raise ValueError(f"Frame length ({len(chunk)}) does not match PC length ({len(pc_chunk)}).")

                # Compute chunk masks
                chunk_masks = chunk * pc_chunk

                # Accumulate mask sum
                mask_sum += np.sum(chunk_masks, axis=0)

                # Update the frame count
                count += chunk_masks.shape[0]

                # Stop processing if last chunk is reached
                if i + self.chunk_size >= num_frames - n_components:
                    break

            # Compute the mean spatial mask
            mean_mask = mask_sum / count

            # Store the mask
            spatial_masks.append(mean_mask)

        spatial_masks = np.array(spatial_masks)
        return spatial_masks


    def _standardize_mean_mask(self, mean_mask: np.ndarray) -> np.ndarray:
        """
        Standardizes the mean mask by subtracting its mean and dividing by its standard deviation.

        Args:
            mean_mask (numpy.ndarray): The input 2D mask to be standardized.

        Returns:
            numpy.ndarray: Standardized mean mask.

        Raises:
            ValueError: If standard deviation is zero, leading to division errors.
        """
        if not isinstance(mean_mask, np.ndarray):
            raise TypeError("mean_mask must be a NumPy array.")

        mean = mean_mask.mean()
        std = mean_mask.std()

        # Validate standard deviation
        if std == 0:
            raise ValueError("Standard deviation is zero, leading to division errors!")

        # Standardize the mean mask
        mean_mask = (mean_mask - mean) / std

        # Check for NaN values after standardization
        nan_count = np.isnan(mean_mask).sum()
        if nan_count > 0:
            logger.warning(f"Standardized mean mask contains {nan_count} NaN values.")

        return mean_mask


    def _plot_spatial_masks(self) -> plt.Figure:
        """
        Plots spatial masks corresponding to principal components.

        Returns:
            matplotlib.figure.Figure: The figure containing the spatial masks.

        Raises:
            AttributeError: If `self.spatial_masks` is missing or None.
        """
        if not hasattr(self, 'spatial_masks') or self.spatial_masks is None:
            raise AttributeError("spatial_masks attribute is missing or None. Run PCA before plotting.")

        if not isinstance(self.spatial_masks, np.ndarray):
            raise TypeError("spatial_masks must be a NumPy array.")

        n_components = self.n_to_plot
        fig,axes = plt.subplots(figsize=(3 * (n_components + 1), 3), nrows = 1, ncols = 4)  # Extra space for the example frame

        # Plot the example frame
        ax = fig.add_subplot(1, n_components + 1, 1)
        im = ax.imshow(self.example_frame, cmap='gray', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        ax.set_title(f'Example Frame', fontsize=10)

        for i, (ax, mask) in enumerate(zip(axes[1:], self.spatial_masks)):
            im = ax.imshow(mask, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            ax.set_title(f'PC {i + 1} mask')

        plt.tight_layout()
        plt.show()

        return fig


    def _plot_explained_variance(self) -> plt.Figure:
        """
        Plots the explained variance ratio of each principal component from a PCA model.

        Returns:
            matplotlib.figure.Figure: The figure containing the explained variance plot.

        Raises:
            ValueError: If PCA model has not been fitted.
        """
        if not hasattr(self.pca, 'explained_variance_ratio_'):
            raise ValueError("PCA object must be fitted before plotting.")

        fig, ax = plt.subplots(figsize=(4, 3))
        explained_variance = self.pca.explained_variance_ratio_ * 100

        ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2, markersize=5)
        ax.set_title('Variance Explained by Principal Components', fontsize=12)
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance (%)', fontsize=12)
        ax.set_xlim([0, 30])  # Show only top 30 components
        plt.tight_layout()
        plt.show()

        return fig


    def _plot_pca_components_traces(self, component_indices: list = [0, 1, 2], remove_outliers: bool = False, axes=None) -> plt.Figure:
        """
        Plots PCA components against time.

        Args:
            component_indices (list, optional): Indices of PCA components to plot. Defaults to [0, 1, 2].
            remove_outliers (bool, optional): Whether to remove outliers above 99%. Defaults to True.
            axes (matplotlib.axes.Axes, optional): Axes for plotting. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The figure containing PCA component traces.

        Raises:
            ValueError: If PCA motion energy data is missing or has fewer than three components.
            AssertionError: If timestamps and PCA traces have different lengths.
        """
        if not hasattr(self, 'pca_motion_energy') or self.pca_motion_energy is None:
            raise ValueError("PCA motion energy data is missing. Run PCA before plotting.")

        if self.pca_motion_energy.shape[1] < 3:
            raise ValueError("pca_motion_energy must have at least 3 components to plot.")

        fps = self.video_metadata.get('fps', 60)  # Default FPS to 60 if missing
        x_range = min(10000, self.pca_motion_energy.shape[0])  # Ensure range doesn't exceed available data
        x_trace_seconds = np.round(np.arange(0, x_range) / fps, 2)

        if axes is None:
            fig, axes = plt.subplots(len(component_indices), 1, figsize=(15, 2 * len(component_indices)))

        for i, ax in enumerate(axes):
            pc_trace = self.pca_motion_energy[:x_range, component_indices[i]]
            if remove_outliers:
                pc_trace = utils.remove_outliers_99(pc_trace)

            assert len(x_trace_seconds) == len(pc_trace), "Timestamps and PC trace lengths do not match."

            ax.plot(x_trace_seconds, pc_trace)
            ax.set_ylabel(f'PCA {component_indices[i] + 1}', fontsize=16)
            ax.set_title(f'PCA {component_indices[i] + 1} over time (s)', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True)

        axes[-1].set_xlabel('Time (s)', fontsize=16)
        plt.tight_layout()
        return fig

    def _get_motion_energy_trace(self):
        npz_data = np.load(self.npz_path)

        if not npz_data:
            raise ValueError("No data found in the NPZ file.")
        
        if self.use_cropped_frames:
            array = npz_data['cropped_frame_motion_energy']
        else:
            array = npz_data['full_frame_motion_energy']

        self.motion_energy_trace = array
        return self
        

    def _plot_motion_energy_trace(self, remove_outliers: bool = True) -> plt.Figure:
        """
        Creates a figure and plots a NumPy array from an NPZ file.

        Args:

        Returns:
            plt.Figure: The matplotlib figure object.

        Raises:
            ValueError: If the specified array name is not found.
        """

        if remove_outliers:
            array = utils.remove_outliers_99(self.motion_energy_trace)
        else:
            array = self.motion_energy_trace

        fig, ax = plt.subplots(figsize=(15, 6))
        if self.use_cropped_frames or self.recrop:
            array_name = 'cropped frames'
        else:
            array_name = 'full frames'


        ax.plot(array, label=f"{array_name}")
        ax.set_title(f"Motion energy trace for {array_name} from NPZ")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        logger.info(f"Plotted array: {array_name}")

        return fig  


    def _save_results(self) -> None:
        """
        Saves the PCA object and results as a pickle file.

        The function constructs the results folder path, creates directories if needed,
        and saves the object as a serialized dictionary.

        Returns:
            None
        """
        logger.info("Saving PCA results...")

        # Construct results folder path
        top_results_folder = utils.construct_results_folder(self.video_metadata)
        self.top_results_path = os.path.join(utils.get_results_path(), top_results_folder)

        # Ensure directory exists before saving
        os.makedirs(self.top_results_path, exist_ok=True)

        # Save the object as a pickle file
        save_path = os.path.join(self.top_results_path, "fit_motion_energy_pca.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(utils.object_to_dict(self), f)

        logger.info(f"PCA results saved at: {save_path}")
        return self
