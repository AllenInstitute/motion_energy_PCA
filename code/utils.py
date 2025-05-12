import os
import json
import pickle
import cv2
import numpy as np
from tqdm import tqdm
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


def construct_results_folder(metadata) -> str:
    """
    Construct a folder name for results storage based on video metadata.

    Returns:
        str: Constructed folder name.

    Raises:
        KeyError: If required metadata fields are missing.
    """
   
    try:
        return f"{metadata['data_asset_name']}_{metadata['camera_label']}_PCA"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def object_to_dict(obj):
    """
    Recursively convert an object (with __dict__) to a dictionary,
    converting any non-JSON-serializable elements to serializable types.
    
    Args:
        obj: The object or structure to convert.
    
    Returns:
        A JSON-serializable dictionary or list representation.
    """
    if hasattr(obj, "__dict__"):
        data = {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, dict):
        data = {key: object_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        data = [object_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        data = tuple(object_to_dict(item) for item in obj)
    else:
        data = obj

    return _obj_to_dict(data)


def _obj_to_dict(data):
    """
    Recursively convert non-JSON-serializable items to JSON-compatible types.
    
    Args:
        data: The data structure (dict, list, etc.) to convert.
    
    Returns:
        A structure with all values converted to JSON-serializable types.
    """
    if isinstance(data, dict):
        return {k: _obj_to_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_obj_to_dict(item) for item in data]
    elif isinstance(data, tuple):
        return [_obj_to_dict(item) for item in data]  # Convert tuple to list
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data
