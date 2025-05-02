import os
import json
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
import logging
import utils  # assumes validate_frame, object_to_dict, construct_results_folder, get_results_folder are in utils.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS = "/results"
class PCAgenerator:
    """
    PCA Generator for motion energy data from video files.
    Applies PCA in chunks with optional standardization and computes spatial masks.
    """

    def __init__(self, video_path: str, standardize4PCA: bool = False):
        """
        Initialize the PCAgenerator object.

        Args:
            video_path (str): Path to the video file (MP4).
            standardize4PCA (bool): Whether to standardize motion energy before PCA.
        """
        self.video_path = Path(video_path)
        self.standardize4PCA = standardize4PCA
        self.n_components = 100
        self.n_to_plot = 3
        self.chunk_size = 300
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from a JSON file located next to the video."""
        json_path = self.video_path.parent / "metadata.json"
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        self.video_metadata = metadata["video_metadata"]
        self.me_metadata = {k: v for k, v in metadata.items() if k != "video_metadata"}
        logger.info("Metadata loaded successfully.")

    @staticmethod
    def _standardize_chunk(chunk: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (chunk - mean) / std

    def _compute_mean_std(self, num_frames: int) -> tuple:
        """Compute mean and std over motion energy frames from the video."""
        cap = cv2.VideoCapture(str(self.video_path))
        means, stds, counts = [], [], []

        for _ in tqdm(range(0, num_frames, self.chunk_size), desc="Computing mean/std"):
            frames = []
            for _ in range(self.chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = utils.validate_frame(frame)
                frames.append(gray.flatten())
            if frames:
                chunk_arr = np.stack(frames)
                means.append(chunk_arr.mean(axis=0))
                stds.append(chunk_arr.std(axis=0))
                counts.append(len(frames))

        cap.release()
        mean = np.average(means, axis=0, weights=counts)
        std = np.average(stds, axis=0, weights=counts)
        std[std == 0] = 1  # avoid division by zero
        return mean, std

    def _apply_pca_to_motion_energy(self):
        """Fit IncrementalPCA to video motion energy frames."""
        cap = cv2.VideoCapture(str(self.video_path))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ipca = IncrementalPCA(n_components=self.n_components)

        mean, std = None, None
        if self.standardize4PCA:
            mean, std = self._compute_mean_std(self.num_frames)

        current_frame = 0
        while current_frame < self.num_frames:
            frames = []
            for _ in range(self.chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = utils.validate_frame(frame)
                frames.append(gray.flatten())

            if not frames:
                break

            chunk = np.stack(frames)
            if self.standardize4PCA:
                chunk = self._standardize_chunk(chunk, mean, std)

            ipca.partial_fit(chunk)
            current_frame += len(frames)

        logger.info("PCA fitting complete.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        transformed_chunks = []
        current_frame = 0

        while current_frame < self.num_frames:
            frames = []
            for _ in range(self.chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = utils.validate_frame(frame)
                frames.append(gray.flatten())

            if not frames:
                break

            chunk = np.stack(frames)
            if self.standardize4PCA:
                chunk = self._standardize_chunk(chunk, mean, std)

            transformed_chunks.append(ipca.transform(chunk))
            current_frame += len(frames)

        cap.release()
        self.ipca = ipca
        self.pca_motion_energy = np.vstack(transformed_chunks)
        logger.info(f"PCA transformation complete. Output shape: {self.pca_motion_energy.shape}")
        return self

    def _compute_spatial_masks(self) -> np.ndarray:
        """
        Computes spatial masks for each principal component.
        Returns:
            np.ndarray: Spatial masks, and the average motion energy frame.
        """
        if self.pca_motion_energy.ndim != 2:
            raise ValueError("pca_motion_energy must be 2D")

        cap = cv2.VideoCapture(str(self.video_path))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n_components = min(self.n_to_plot, self.pca_motion_energy.shape[1])
        spatial_masks = []

        logger.info(f"Number of PCA components: {self.pca_motion_energy.shape[1]}")

        # Compute mean frame for optional visualization
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        mean_frames = []
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break
            gray = utils.validate_frame(frame)
            mean_frames.append(gray)
        self.mean_me_mask = np.mean(np.stack(mean_frames), axis=0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for pc_index in range(n_components):
            logger.info(f"Processing Principal Component {pc_index + 1}")
            pc = self.pca_motion_energy[:, pc_index]
            mask_sum = np.zeros((frame_height, frame_width), dtype=np.float64)
            count = 0
            current_index = 0

            while current_index < len(pc):
                frames = []
                pc_chunk = []

                for _ in range(self.chunk_size):
                    ret, frame = cap.read()
                    if not ret or current_index >= len(pc):
                        break
                    gray = utils.validate_frame(frame)
                    frames.append(gray)
                    pc_chunk.append(pc[current_index])
                    current_index += 1

                if not frames:
                    break

                frames_arr = np.stack(frames)
                pc_arr = np.array(pc_chunk)[:, np.newaxis, np.newaxis]
                mask_sum += np.sum(frames_arr * pc_arr, axis=0)
                count += len(frames)

            spatial_masks.append(mask_sum / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()
        return np.array(spatial_masks)

    def _add_spatial_masks(self) -> None:
        """Computes and adds spatial masks to the object."""
        logger.info("Computing spatial masks...")
        self.spatial_masks = self._compute_spatial_masks()
        logger.info("Spatial masks computation complete.")
        return self

    def _save_results(self) -> None:
        """
        Saves the PCAgenerator metadata as JSON and the fitted PCA model as a pickle file.

        JSON includes all serializable metadata from the object (excluding arrays and models).
        Pickle stores the fitted PCA model.
        """
        logger.info("Saving PCA results...")

        # Construct results path
        top_results_folder = utils.construct_results_folder(self.video_metadata)
        self.top_results_path = os.path.join(RESULTS, top_results_folder)
        os.makedirs(self.top_results_path, exist_ok=True)

        # Save serializable attributes as JSON
        json_dict = {
            "video_path": str(self.video_path),
            "n_components": self.n_components,
            "n_to_plot": self.n_to_plot,
            "chunk_size": self.chunk_size,
            "standardize4PCA": self.standardize4PCA,
            "video_metadata": self.video_metadata,
            "me_metadata": self.me_metadata,
            "top_results_path": self.top_results_path,
            "pcs": self.pca_motion_energy,
        }
        json_path = os.path.join(self.top_results_path, "pca_generator_metadata.json")
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
        logger.info(f"PCA metadata saved to: {json_path}")

        # Save PCA model as pickle
        try:
            pkl_path = os.path.join(self.top_results_path, "pca_model.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self.ipca, f)
            logger.info(f"PCA model saved to: {pkl_path}")
        except Exception as e:
            logger.warning(f"Could not save PCA model: {e}")

        return self
