# PCA-Based Motion Energy Analysis  

This capsule is a part of Behavior Video QC to apply Principal Component Analysis (PCA) to motion energy data. The script loads motion energy data, applies PCA, generates spatial masks, and visualizes the results. Computing time depends on number and size of frames. An hour long video with ~600x400 frames at 60 Hz takes about 30 min.

## Overview

`PCAgenerator` is a Python class designed to apply Incremental PCA on motion energy data extracted from video files (e.g., MP4 format). It processes large videos efficiently by loading them in chunks, applying PCA, and generating interpretable spatial masks for each component.

This tool is optimized for neuroscience video experiments (e.g., mouse behavior or facial movement) where dimensionality reduction and visual inspection of motion structure are essential.

---

## Features

- ✅ Incremental PCA for large video files (doesn't load all frames into memory)
- ✅ Optional per-frame standardization
- ✅ Computation of spatial masks (PCA projection × motion energy frame)
- ✅ Automatic metadata loading and structured saving of results
- ✅ Clean JSON + Pickle output format

---

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- tqdm

---

## Usage

### 1. Prepare Your Video Files

Store MP4 video files in the `/data/...` directory. Each should end in `motion_energy.mp4`.

### 2. Run Batch Processing

```bash
python run.py
