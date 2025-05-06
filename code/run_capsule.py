
from tqdm import tqdm
import time  # Added for timing
from PCAgenerator import PCAgenerator
from pathlib import Path

DATA_PATH = Path("/data")
video_paths = list(DATA_PATH.glob("*/*/*motion_energy.mp4"))
print(f"found {len(video_paths)} motion energy videos")
def run():
    for video_path in tqdm(video_paths):
        print(f'...Loading {video_path}')
        
        start_time = time.time()  # Start the timer

        me_pca = PCAgenerator(video_path = video_path, standardize4PCA=False) 
        
        me_pca._apply_pca_to_motion_energy()

        me_pca._add_spatial_masks()

        me_pca._save_results()
        
        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

if __name__ == "__main__": 
    run()