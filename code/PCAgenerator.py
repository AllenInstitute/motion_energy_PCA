import numpy as np
import zarr
import dask
import dask.array as da
import json
import os
import utils

class PCAgenerator:
    def __init__(self, motion_zarr_path: str):
        self.frame_zarr_path = frame_zarr_path
        self.zarr_store_frames = zarr.DirectoryStore(frame_zarr_path)
        self.loaded_metadata = None