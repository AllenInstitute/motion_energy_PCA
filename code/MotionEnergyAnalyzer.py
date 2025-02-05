
import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils
import pickle

class MotionEnergyAnalyzer:
    def __init__(self, frame_zarr_path: str):
        self.frame_zarr_path = frame_zarr_path
        self.zarr_store_frames = zarr.DirectoryStore(frame_zarr_path)
        self.loaded_metadata = None

    def _load_metadata(self):
        """Load metadata from the Zarr store."""
        root_group = zarr.open_group(self.zarr_store_frames, mode='r')
        metadata = json.loads(root_group.attrs['metadata'])
        metadata['crop'] = False
        self.loaded_metadata = metadata
        

    ## TypeError: _compute_motion_energy() takes 1 positional argument but 2 were given

    # def _compute_motion_energy(frames):
    #     """
    #     Compute motion energy from a set of frames.
    #     Motion energy is computed as the sum of absolute differences between consecutive frames.
    #     """
    #     if len(frames) < 2:
    #         raise ValueError("At least two frames are required to compute motion energy.")
        
    #     motion_energy = da.abs(frames[1:] - frames[:-1])
    #     return motion_energy


    def analyze(self):
        """
        Analyze motion energy based on the frames.
        Applies cropping if the crop attribute is True and saves results.
        """
        # Load the frames from Zarr
        grayscale_frames = da.from_zarr(self.zarr_store_frames, component='data')
        #grayscale_frames = grayscale_frames[:10000,:,:]
        #print('using subset of frames for testing')
        # Load metadata
        self._load_metadata()

        # Check for cropping option
        crop = self.loaded_metadata.get('crop')
        H, W = self.loaded_metadata.get('height'), self.loaded_metadata.get('width')

        if crop:
            crop_region = utils.get_crop_region()
            crop_y_start, crop_x_start, crop_y_end, crop_x_end = crop_region
            motion_energy = da.abs(grayscale_frames[1:] - grayscale_frames[:-1])
            motion_energy = motion_energy.rechunk((100, H, W))  # Adjust based on available memory
            #motion_energy = self._compute_motion_energy(grayscale_frames)
            cropped_motion_energy = motion_energy[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            H, W = np.diff(crop_y_end, crop_y_start), np.diff(crop_x_end, crop_x_start)
            cropped_motion_energy = cropped_motion_energy.rechunk((100, H, W)) 
        else:
            motion_energy = da.abs(grayscale_frames[1:] - grayscale_frames[:-1])
            motion_energy = motion_energy.rechunk((100, H, W)) 
            #motion_energy = self._compute_motion_energy(grayscale_frames)


        # Save motion energy frames as a video
        # path in results to where data from this video will be saved
        top_zarr_folder = utils.construct_zarr_folder(self.loaded_metadata)
        top_zarr_path = os.path.join(utils.get_results_path(), top_zarr_folder)

        utils.save_video(frames = motion_energy, video_path = top_zarr_path, fps=self.loaded_metadata.get('fps'), num_frames=100)

        # Save motion energy frames to Zarr
        me_zarr_path = utils.get_zarr_path(self.loaded_metadata, path_to='motion_energy')
        print('Creating folder for motion energy frames of full video...')
        me_zarr_store = zarr.DirectoryStore(me_zarr_path)
        # Perform operations with the Zarr store
        root_group = zarr.group(me_zarr_store, overwrite=True)
        motion_energy.to_zarr(me_zarr_store, component='data', overwrite=True)
        if crop:
            cropped_motion_energy.to_zarr(me_zarr_store, component='cropped_data', overwrite=True)

        # Add metadata to the Zarr store
        root_group.attrs['metadata'] = json.dumps(self.loaded_metadata)
        print(f'Saved motion energy frames to {me_zarr_path}')

        # Compute trace and save it to the object
        sum_trace = motion_energy.sum(axis=(1, 2)).compute()
        self.motion_energy_sum = sum_trace.reshape(-1, 1)

        #save object
        with open(f'{top_zarr_path}/{top_zarr_folder}.pkl', 'wb') as file:
            pickle.dump(self, file)
        print('saved object')

        # save motion energy trace for redundancy as np array
        np.savez(f'{top_zarr_path}/{top_zarr_folder}.npz', array1 = self.motion_energy_sum)
        print('saved me trace')

        

