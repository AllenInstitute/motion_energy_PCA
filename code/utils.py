
import os
import json
import pickle
import numpy as np
#from MotionEnergyAnalyzer import MotionEnergyAnalyzer

def find_files(root_dir: str, endswith = '', return_dir = True):
    """
    Recursively search for all Zarr files in the specified root directory
    and save their paths to a JSON file.

    Args:
        root_dir (str): Root directory to search for Zarr files.

    Returns:
        list: List of paths to the found Zarr files.
    """
    collected_files = []

    if return_dir:
        # Walk through the directory tree
        for root, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                if dir_name.endswith(endswith):
                    collected_files.append(os.path.join(root, dir_name))
    else: #return files
        for root, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith(endswith): 
                    collected_files.append(os.path.join(root, file_name))

    # # Save the paths to a JSON file
    # with open(output_file, "w") as file:
    #     json.dump(files, file, indent=4)

    return collected_files


# def check_crop_region(pkl_file : str):
#     meta_obj = load()
#     if hasattr(meta_obj, 'crop_region'):
#         crop_region = meta_obj['crop_region']
#     else:
#         crop_region = None
#     return crop_region



def load_pickle_file(file_path: str):
    """
    Load a pickle file from the given path and return the deserialized object.

    Parameters:
        file_path (str): The path to the pickle file.

    Returns:
        object: The Python object deserialized from the pickle file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid pickle file or is corrupted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
    
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error loading pickle file: {e}")


def get_x_trace_sec(me_frames, fps=60):
    x_trace_seconds = np.round(np.arange(1, frames.shape[0]) / fps, 2)
    return x_trace_seconds

def get_results_path() -> str:
    """
    Retrieve the path to the results folder. Modify this function as needed to fit your project structure.

    Returns:
        str: Path to the results folder.
    """
    # Placeholder implementation, update with actual results folder logic if needed
    return '/root/capsule/results'

def construct_results_folder(metadata: dict) -> str:
    """
    Construct the folder name for results storage based on metadata.

    Args:
        metadata (dict): A dictionary containing 'mouse_id', 'camera_label', and 'data_asset_id'.

    Returns:
        str: Constructed folder name.
    """
    try:
        return f"{metadata['mouse_id']}_{metadata['camera_label']}_{metadata['data_asset_id']}"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def remove_outliers_99(arr):
    """
    Removes outliers above the 99th percentile in a NumPy array.

    Parameters:
    - arr (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: Array with outliers removed (values above 99th percentile).
    """
    threshold = np.percentile(arr, 99)  # Compute the 99th percentile
    return arr[arr <= threshold]  # Keep only values below or equal to the threshold

    print(f"Original size: {data.size}, Filtered size: {filtered_data.size}")

    import matplotlib.pyplot as plt
import os

def save_figure(fig, save_path, fig_name, dpi=300, bbox_inches="tight", transparent=True):
    """
    Saves a Matplotlib figure to the specified path.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure to save.
    - save_path (str): The file path (including extension) to save the figure.
    - fig_name (str)
    - dpi (int, optional): Dots per inch (default: 300 for high resolution).
    - bbox_inches (str, optional): Bounding box to trim white space (default: "tight").
    - transparent (bool, optional): Whether to save with a transparent background (default: True).
    
    Returns:
    - None
    """
    figpath = save_path+'/'+fig_name
    os.makedirs(os.path.dirname(figpath), exist_ok=True)  # Ensure directory exists
    fig.savefig(figpath, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    print(f"Figure saved at: {save_path}")

def object_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    else:
        return obj

####################### OLD FUNCTIONS

def get_zarr_path(metadata: dict, path_to: str = 'motion_energy') -> str:
    """
    Construct the path for saving Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing metadata such as 'mouse_id', 'camera_label', and 'data_asset_id'.
        path_to (str): Specifies the type of frames to be saved ('gray_frames' or 'motion_energy_frames').

    Returns:
        str: Full path to the Zarr storage file.
    """
    zarr_folder = construct_zarr_folder(metadata)
    zarr_path = os.path.join(get_results_folder(), zarr_folder)

    # Create the directory if it doesn't exist
    os.makedirs(zarr_path, exist_ok=True)

    filename = 'processed_frames.zarr' if path_to == 'gray_frames' else 'motion_energy_frames.zarr'
    return os.path.join(zarr_path, filename)

def get_data_path(metadata: dict) -> str:
    """
    Construct the path for data storage based on metadata.

    Args:
        metadata (dict): A dictionary containing metadata such as 'mouse_id', 'camera_label', and 'data_asset_id'.

    Returns:
        str: Full path to the data storage folder.
    """
    data_folder = construct_zarr_folder(metadata)
    data_path = os.path.join(get_results_folder(), data_folder)

    # Create the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    return data_path


def construct_zarr_folder(metadata: dict) -> str:
    """
    Construct the folder name for Zarr storage based on metadata.

    Args:
        metadata (dict): A dictionary containing 'mouse_id', 'camera_label', and 'data_asset_id'.

    Returns:
        str: Constructed folder name.
    """
    try:
        return f"{metadata['mouse_id']}_{metadata['camera_label']}_{metadata['data_asset_id']}"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")

def get_fps(file_path):
    meta = load_pickle_file(file_path)
    for key, value in meta.loaded_metadata.items():
        if "fps" in key.lower():
            print(f"Found key: '{key}' with value: {value}")
            return value
    else:
        print("fps not found.")
    
    return None