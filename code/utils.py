
import os
import json



def get_results_folder() -> str:
    """
    Retrieve the path to the results folder. Modify this function as needed to fit your project structure.

    Returns:
        str: Path to the results folder.
    """
    # Placeholder implementation, update with actual results folder logic if needed
    return '/root/capsule/results'


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


def save_video(frames, video_path = '', video_name='motion_energy_clip.avi', fps=60, num_frames = 1000):
    """
    Save the provided frames to a video file using OpenCV.
    """
    
    output_video_path = os.path.join(video_path, video_name)
    # Get frame shape and number of frames
    print(type(frames))
    _, frame_height, frame_width = frames.shape

    # Specify the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Process and write each frame to the video file
    for i in range(5000, num_frames+5000):
        frame = frames[i].compute()  # Compute the frame to load it into memory
        frame = frame.astype('uint8')  # Ensure the frame is of type uint8 for video
        out.write(frame)  # Write the frame to the video file

    # Release the video writer
    out.release()
    print(f"Video saved to '{output_video_path}'")