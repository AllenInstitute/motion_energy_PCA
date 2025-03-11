import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def get_data_path(pipeline: bool = True) -> Path:
    """
    Get the data folder path.

    Returns:
        str: Path to the results folder.
    """
    if pipeline:
        return Path('/data/')
    else:
        return Path('/root/capsule/data')


def get_results_folder(pipeline: bool = True) -> Path:
    """
    Get the results folder path.

    Returns:
        str: Path to the results folder.
    """
    if pipeline:
        return Path('/results/')
    else:
        return Path('/root/capsule/results')

import os


def find_input_paths(directory: Path = Path(), return_file = False, tag: str = '', endswith = '') -> list:
    """
    Retrieve paths to Zarr directories within the specified directory, optionally filtered by a subdirectory.

    Args:
        directory (Path): The base directory to search for Zarr files.
        tag (str): str tag in video filename to include. (not being used)

    Returns:
        list: A list of paths to Zarr directories.
    """
    input_paths = []
    for root, dirs, files in os.walk(directory):
        if return_file is False:
            for d in tqdm(dirs, desc=f"Searching for Zarr directories in {root}"):
                #print(f'.....directory {d}.....')
                if endswith in d:
                    full_path = os.path.join(root, d)
                    print(f"\n.  Found {endswith} directory: {full_path}")
                    input_paths.append(full_path)
        else:
            for f in tqdm(files, desc=f"Searching for files in {root}"):
                #print(f'.....file {f}.....')
                if endswith in f:
                    full_path = os.path.join(root, f)
                    print(f"\n.  Found {endswith} file: {full_path}")
                    input_paths.append(full_path)
   
    return input_paths


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


def construct_results_folder(metadata: dict) -> str:
    """
    Construct a folder name for results storage based on metadata.

    Args:
        metadata (dict): Dictionary containing 'mouse_id', 'camera_label', and 'data_asset_name'.

    Returns:
        str: Constructed folder name.

    Raises:
        KeyError: If required metadata fields are missing.
    """
    try:
        return f"{metadata['mouse_id']}_{metadata['data_asset_name']}_{metadata['camera_label']}_PCA"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")

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




# def find_files(directory: Path, endswith: str ) -> list:
#     return [
#         str(p) for p in directory.rglob(endswith)
#     ]

# def find_files_old(root_dir: Path, endswith: str = '', return_dir: bool = True) -> list:
#     """
#     Recursively search for files or directories ending with a specific string in a given root directory.

#     Args:
#         root_dir (str): Root directory to search.
#         endswith (str, optional): Suffix to filter files or directories. Defaults to '' (no filtering).
#         return_dir (bool, optional): If True, returns directories. If False, returns files. Defaults to True.

#     Returns:
#         list: List of paths to the found files or directories.
#     """
#     collected_files = []

#     for root, dirs, files in os.walk(root_dir):
#         print(f'{root}, {dirs}, {files}')
#         if return_dir:
#             for d in tqdm(dirs, desc=f"Searching for Zarr directories in {root}"):
#                 if d.endswith(endswith):
#                     collected_files.append(os.path.join(root, d))
#         else:
#             for f in tqdm(files, desc=f"Searching for Zarr files in {root}"):
#                 if f.endswith(endswith):
#                     collected_files.append(os.path.join(root, f))

#     return collected_files