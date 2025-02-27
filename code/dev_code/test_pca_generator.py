import pytest
import numpy as np
import os
import pickle
from pca_generator import PCAgenerator  # Import your class


@pytest.fixture
def mock_pca():
    """Fixture to create a PCAgenerator instance for testing."""
    return PCAgenerator(motion_zarr_path="test.zarr", crop=True, crop_region=(10, 20, 50, 60))


def test_standardize_chunk(mock_pca):
    """Test standardization of data."""
    data = np.random.rand(5, 10)  # Small random dataset
    standardized_data = mock_pca._standardize_chunk(data, 1)
    assert np.allclose(standardized_data.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(standardized_data.std(axis=0), 1, atol=1e-5)


def test_crop_frames(mock_pca):
    """Test frame cropping."""
    frames = np.random.rand(100, 100, 100)  # Fake 3D frames
    cropped_frames, height, width = mock_pca._crop_frames(frames)
    assert cropped_frames.shape == (100, 40, 40)  # Check correct cropping
    assert height == 40
    assert width == 40


def test_save_results(mock_pca, tmpdir):
    """Test saving PCA results."""
    mock_pca.top_results_path = str(tmpdir)  # Use temp directory
    mock_pca._save_results()
    
    save_path = os.path.join(mock_pca.top_results_path, "fit_motion_energy_pca.pkl")
    assert os.path.exists(save_path)

    with open(save_path, "rb") as f:
        saved_object = pickle.load(f)
        assert isinstance(saved_object, dict)  # Ensure it's saved as a dictionary
