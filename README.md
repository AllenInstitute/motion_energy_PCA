# PCA-Based Motion Energy Analysis  

This capsule is a part of Behavior Video QC to apply Principal Component Analysis (PCA) to motion energy data extracted from Zarr files. The script loads motion energy data, applies PCA, generates spatial masks, and visualizes the results. Computing time depends on number and size of frames. An hour long video with ~600x400 frames at 60 Hz takes about 30 min.

## Features  
- Loads motion energy data from Zarr and NPZ files.  
- Applies PCA to extract meaningful components from motion energy.  
- Supports optional cropping of motion energy frames before PCA.  
- Generates and saves spatial masks for PCA components.  
- Saves results, including PCA components, explained variance, and motion energy traces.  
- Tracks processing time.  

## Prerequisites  

Ensure you have the following dependencies installed:  

```bash
pip install tqdm numpy zarr matplotlib
```

## Usage  

### Running the script  
Execute the script using:  
```bash
python run_capsule.py
```

### Parameters  

| Parameter            | Description                                              | Default Value               |
|----------------------|----------------------------------------------------------|-----------------------------|
| `zarr_paths`        | List of paths to Zarr directories containing motion energy data. | `utils.find_files()`|
| `npz_paths`         | List of paths to NPZ files storing motion energy traces. | `utils.find_files()` |
| `crop`              | Boolean flag to enable cropping before PCA. Checks ME metadata.  | `None`                      |
| `crop_region`       | Tuple defining the crop region `(y_start, x_start, y_end, x_end)`. | `checks metadata` |
| `standardize4PCA`   | Boolean flag for standardizing data before PCA.          | `False`                     |
| `standardizeMasks`  | Boolean flag for standardizing spatial masks plots.      | `True`                      |


### Example Output

```
Processing /root/capsule/data/video1.zarr
Applying PCA to motion energy data...
Saving PCA results...
Generating and saving spatial masks...
Plotting and saving PCA explained variance...
Plotting and saving motion energy trace...
Plotting and saving PCA component traces...
Total time taken: 360.45 seconds
```
### Modifying Parameters  

To process different Zarr and NPZ files, modify the `root_dir` parameter in `script.py`:  

```python
zarr_paths = utils.find_files(root_dir='/your/custom/path', endswith='zarr')
npz_paths = utils.find_files(root_dir='/your/custom/path', endswith='.npz', return_dir=False)
```

To change the cropping region, update the `crop_region` variable:
```python
crop_region = (y_start, x_start, y_end, x_end)  # Adjust cropping dimensions in pixel values
```

To enable or disable standardization before PCA, update these parameters:

```python
me_pca = PCAgenerator(zarr_path, npz_path, crop=True, crop_region=crop_region, 
                      standardize4PCA=True, standardizeMasks=False)
```



