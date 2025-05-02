import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os

def validate_frame(frame: np.ndarray) -> np.ndarray:
    """
    Ensure the input frame is grayscale. If not, convert it to grayscale.

    Args:
        frame (np.ndarray): Input video frame (BGR or grayscale).

    Returns:
        np.ndarray: Grayscale frame.

    Raises:
        ValueError: If frame is not a 2D (grayscale) or 3D (color) NumPy array.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if frame.ndim == 2:
        # Already grayscale
        return frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        # Convert BGR to grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}. Expected 2D grayscale or 3D color frame.")


def construct_results_folder(self) -> str:
    """
    Construct a folder name for results storage based on video metadata.

    Returns:
        str: Constructed folder name.

    Raises:
        KeyError: If required metadata fields are missing.
    """
    metadata = self.video_metadata
    try:
        return f"{metadata['data_asset_name']}_{metadata['camera_label']}_PCA"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")

def object_to_dict(obj):
    if hasattr(obj, "__dict__"):
        meta_dict = {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        meta_dict = [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        meta_dict = {key: object_to_dict(value) for key, value in obj.items()}
    else:
        meta_dict = obj

    # Convert Path to str for json serialization
    if isinstance(meta_dict, dict):
        return {k: str(v) if isinstance(v, Path) else v for k, v in meta_dict.items()}
    elif isinstance(meta_dict, list):
        return [str(v) if isinstance(v, Path) else v for v in meta_dict]
    elif isinstance(meta_dict, Path):
        return str(meta_dict)
    else:
        return meta_dict

