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

    def __init__(self, motion_zarr_path: str, pkl_file: str = None, npz_file: str = None, use_cropped_frames: bool = True, recrop: bool = None, crop_region: tuple = None,
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
        self.pkl_file = pkl_file
        self.npz_file = npz_file
        self.recrop = recrop
        self.crop_region = crop_region
        self.use_cropped_frames = use_cropped_frames
        self.n_components = 100  # Number of PCA components
        self.n_to_plot = 3  # Number of components to visualize
        self.standardize4PCA = standardize4PCA
        self.chunk_size = 100
        self.start_index = 0  # First frame with data info should have been dropped when me was computed
        self.mean = None
        self.std = None
        self._load_metadata()
        self.me_metadata['crop'] = True # find why it was saved as False
        #self._compare_crop_settings()
        #self._get_motion_energy_trace()

    
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
        metadata_str = root_group.attrs.get('metadata', None)
        if metadata_str is None:
            print("Metadata attribute not found in root_group.attrs. Will load metadata via pkl file.")
            try:
                all_metadata = utils.load_pickle_file(self.pkl_file)
                self.video_metadata = all_metadata.get('video_metadata')
                me_metadata = all_metadata.pop('video_metadata',None) #remove video metadata
                self.me_metadata = me_metadata
                logger.info("Metadata loaded successfully.")
            except TypeError:
                print(f'no pickle file in this dataset {self.zarr_file_path}')
                print('Will try to run without metadata')
                self.video_metadata = {}
                self.me_metadata = {}
        else:
            all_metadata = json.loads(metadata_str)
            self.video_metadata = all_metadata.get('video_metadata')
            me_metadata = all_metadata.pop('video_metadata',None) #remove video metadata
            self.me_metadata = me_metadata
            logger.info("Metadata loaded successfully.")

        return self

    def _define_crop_region(self, crop_region: tuple = None) -> None:
        """Define and validate the crop region."""
        if crop_region is None:
            crop_region = self.video_metadata.get('crop_region', (200, 290, 280, 360))
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
        self.explained_variance = ipca.explained_variance_ratio_

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

        
        self.mean_me_frame = np.mean(post_crop_frames_me[100:200], axis=0)
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
        self.top_results_path = os.path.join(utils.get_results_folder(), top_results_folder)

        # Ensure directory exists before saving
        os.makedirs(self.top_results_path, exist_ok=True)

        # Save the object as a pickle file
        save_path = os.path.join(self.top_results_path, "fit_motion_energy_pca.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(utils.object_to_dict(self), f)
        logger.info(f"PCA dictionary saved at: {save_path}")
        try:
            # Save PCA object to a file
            save_path = os.path.join(self.top_results_path, "pca_model.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(self.pca, f)
        except:
            logger.info("Could not save pca object")

        return self
