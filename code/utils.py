import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt



def find_files(root_dir: str, endswith: str = '', return_dir: bool = True) -> list:
    """
    Recursively search for files or directories ending with a specific string in a given root directory.

    Args:
        root_dir (str): Root directory to search.
        endswith (str, optional): Suffix to filter files or directories. Defaults to '' (no filtering).
        return_dir (bool, optional): If True, returns directories. If False, returns files. Defaults to True.

    Returns:
        list: List of paths to the found files or directories.
    """
    collected_files = []

    for root, dirs, files in os.walk(root_dir):
        if return_dir:
            for dir_name in dirs:
                if dir_name.endswith(endswith):
                    collected_files.append(os.path.join(root, dir_name))
        else:
            for file_name in files:
                if file_name.endswith(endswith):
                    collected_files.append(os.path.join(root, file_name))

    return collected_files


def load_pickle_file(file_path: str):
    """
    Load a pickle file and return the deserialized object.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupted or not a valid pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error loading pickle file: {e}")


def get_x_trace_sec(me_frames: np.ndarray, fps: int = 60) -> np.ndarray:
    """
    Generate an x-axis trace in seconds for motion energy frames.

    Args:
        me_frames (numpy.ndarray): Array of motion energy frames.
        fps (int, optional): Frames per second. Defaults to 60.

    Returns:
        numpy.ndarray: Time values in seconds.
    """
    return np.round(np.arange(1, me_frames.shape[0]) / fps, 2)


def get_results_path() -> str:
    """
    Retrieve the path to the results folder.

    Returns:
        str: Path to the results folder.
    """
    return "/root/capsule/results"


def construct_results_folder(metadata: dict) -> str:
    """
    Construct a folder name for results storage based on metadata.

    Args:
        metadata (dict): Dictionary containing 'mouse_id', 'camera_label', and 'data_asset_id'.

    Returns:
        str: Constructed folder name.

    Raises:
        KeyError: If required metadata fields are missing.
    """
    try:
        return f"{metadata['mouse_id']}_{metadata['camera_label']}_{metadata['data_asset_id']}"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def remove_outliers_99(arr: np.ndarray) -> np.ndarray:
    """
    Removes outliers above the 99th percentile in a NumPy array.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Array with outliers removed.
    """
    threshold = np.percentile(arr, 99)
    arr_out = arr.copy()  # Avoid modifying the original array
    arr_out[arr_out > threshold] = np.nan
    return arr_out


def save_figure(fig: plt.Figure, save_path: str, fig_name: str, dpi: int = 300,
                bbox_inches: str = "tight", transparent: bool = True) -> None:
    """
    Save a Matplotlib figure to a specified path.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        save_path (str): Directory where the figure should be saved.
        fig_name (str): Filename of the saved figure.
        dpi (int, optional): Resolution in dots per inch. Defaults to 300.
        bbox_inches (str, optional): Trim white space. Defaults to "tight".
        transparent (bool, optional): Save with transparent background. Defaults to True.
    """
    figpath = os.path.join(save_path, fig_name)
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    fig.savefig(figpath, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    print(f"Figure saved at: {figpath}")


def object_to_dict(obj):
    """
    Recursively converts an object to a dictionary.

    Args:
        obj: The object to convert.

    Returns:
        dict: The dictionary representation of the object.
    """
    if hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    return obj


def get_zarr_path(metadata: dict, path_to: str = "motion_energy") -> str:
    """
    Construct the path for saving Zarr storage based on metadata.

    Args:
        metadata (dict): Metadata containing 'mouse_id', 'camera_label', and 'data_asset_id'.
        path_to (str, optional): Specifies type of frames ('gray_frames' or 'motion_energy_frames'). Defaults to 'motion_energy'.

    Returns:
        str: Full path to the Zarr storage file.
    """
    zarr_folder = construct_zarr_folder(metadata)
    zarr_path = os.path.join(get_results_path(), zarr_folder)
    os.makedirs(zarr_path, exist_ok=True)

    filename = "processed_frames.zarr" if path_to == "gray_frames" else "motion_energy_frames.zarr"
    return os.path.join(zarr_path, filename)


def load_npz_file(npz_file_path: str) -> dict:
    """
    Loads a NumPy `.npz` file and returns its contents.

    Args:
        npz_file_path (str): Path to the `.npz` file.

    Returns:
        dict: Dictionary containing NumPy arrays from the NPZ file.

    Raises:
        FileNotFoundError: If the specified NPZ file does not exist.
        ValueError: If the NPZ file is empty.
    """
    
    npz_data = np.load(npz_file_path)
    return npz_data


    #     try:
    #     if not npz_data.files:
    #         raise ValueError(f"NPZ file {npz_file_path} is empty.")

    #     logger.info(f"Loaded NPZ file: {npz_file_path} | Arrays: {list(npz_data.keys())}")
    #     return {key: npz_data[key] for key in npz_data.files}

    # except FileNotFoundError as e:
    #     logger.error(f"File not found: {npz_file_path}")
    #     raise e
    # except Exception as e:
    #     logger.error(f"Error loading NPZ file: {e}")
    #     raise e

# def get_fps(file_path: str) -> int:
#     """
#     Retrieve the frames per second (FPS) from a metadata pickle file.

#     Args:
#         file_path (str): Path to the pickle file.

#     Returns:
#         int: FPS value if found, otherwise None.
#     """
#     meta = load_pickle_file(file_path)
#     for key, value in meta.loaded_metadata.items():
#         if "fps" in key.lower():
#             print(f"Found key: '{key}' with value: {value}")
#             return value
#     print("FPS not found.")
#     return None
