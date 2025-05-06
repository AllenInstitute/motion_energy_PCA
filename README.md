# PCA-Based Motion Energy Analysis  

This capsule is a part of Behavior Video QC to apply Principal Component Analysis (PCA) to motion energy data. The script loads motion energy data, applies PCA, generates spatial masks, and visualizes the results. Computing time depends on number and size of frames. An hour long video with ~600x400 frames at 60 Hz takes about 30 min.

---

# PCAgenerator

`PCAgenerator` is a Python class for computing **Principal Components Analysis (PCA)** on motion energy data extracted from video files (typically MP4). It is designed to handle long videos by processing frames in **chunks**, with optional standardization, and to compute **spatial masks** for visualization of principal components.

## Features

- Incremental PCA on video motion energy.
- Chunked processing for memory efficiency.
- Optional standardization before PCA.
- Visualization-ready spatial masks for selected principal components.
- Saves PCA metadata and model for reuse and analysis.

## Dependencies

Ensure the following packages are installed:

- Python 3.7+
- numpy
- opencv-python
- scikit-learn
- tqdm

Also requires a custom `utils.py` module with the following functions:
- `validate_frame(frame)`
- `construct_results_folder(metadata)`
- `get_results_folder(metadata)`
- `object_to_dict(obj)`

## Installation

```bash
pip install numpy opencv-python scikit-learn tqdm
```

## Usage

### Step 1: Prepare Your Video

Ensure the video is an `.mp4` file and that a `postprocess_metadata.json` file is present in the same folder. This JSON should include:

```json
{
  "video_metadata": {...},
  "other_keys": {...}
}
```

### Step 2: Run PCA

```python
from pca_generator import PCAgenerator

pca = PCAgenerator(video_path="path/to/video.mp4", standardize4PCA=True)
pca._apply_pca_to_motion_energy()
pca._add_spatial_masks()
pca._save_results()
```

### Output

* PCA-transformed motion energy array: `.pca_motion_energy`
* Spatial masks of principal components: `.spatial_masks`
* Metadata and PCA model are saved in: `/results/{auto_generated_folder}/`

## Output Files

* `pca_generator_metadata.json` — contains all run metadata and parameters.
* `pca_model.pkl` — Pickle file with the trained `IncrementalPCA` model.
* All files are stored under the auto-generated directory in `/results/`.

## Logging

Progress and status messages are logged via Python’s `logging` module at the `INFO` level.

## Notes

* PCA is done using `IncrementalPCA` to handle large datasets.
* If standardization is enabled, PCA is applied to zero-mean, unit-variance data.
* Spatial masks represent how strongly each pixel contributes to selected principal components over time.


---





Store MP4 video files in the `/data/...` directory. Each should end in `motion_energy.mp4`.

### 2. Run Batch Processing

```bash
python run.py
