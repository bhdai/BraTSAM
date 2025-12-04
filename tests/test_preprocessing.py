"""Unit tests for the preprocessing package."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocessing.normalize import normalize_slice
from preprocessing.formats import load_image


class TestNormalizeSlice:
    """Tests for normalize_slice function."""

    def test_normalize_slice_scales_to_255(self):
        """Normal range input should scale to 0-255."""
        # Create a slice with known min/max values
        slice_2d = np.array([[0, 50], [100, 200]], dtype=np.float32)
        result = normalize_slice(slice_2d)

        # Check that output is scaled correctly
        assert result.min() == 0
        assert result.max() == 255

    def test_normalize_slice_handles_constant_image(self):
        """Constant slice should return zeros, not NaN."""
        # Create a constant value slice
        constant_slice = np.full((10, 10), 42.0, dtype=np.float32)
        result = normalize_slice(constant_slice)

        # All values should be zero
        assert np.all(result == 0)
        # Should not contain NaN values
        assert not np.any(np.isnan(result))

    def test_normalize_slice_returns_uint8(self):
        """Output should be uint8 dtype."""
        slice_2d = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
        result = normalize_slice(slice_2d)

        assert result.dtype == np.uint8

    def test_normalize_slice_preserves_shape(self):
        """Output shape should match input shape."""
        slice_2d = np.random.rand(64, 64).astype(np.float32)
        result = normalize_slice(slice_2d)

        assert result.shape == slice_2d.shape

    def test_normalize_slice_handles_negative_values(self):
        """Should correctly normalize slices with negative values."""
        slice_2d = np.array([[-100, -50], [0, 100]], dtype=np.float32)
        result = normalize_slice(slice_2d)

        # Min should map to 0, max to 255
        assert result.min() == 0
        assert result.max() == 255
        assert result.dtype == np.uint8


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_image_loads_png(self):
        """Should load PNG file and return numpy array."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a simple test image
            img_array = np.array([[100, 150], [200, 50]], dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f.name)

            # Load and verify
            result = load_image(f.name)
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 2, 3)  # RGB, even if source was grayscale

            # Clean up
            Path(f.name).unlink()

    def test_load_image_loads_jpg(self):
        """Should load JPG file and return numpy array."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a simple RGB test image
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f.name)

            # Load and verify
            result = load_image(f.name)
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3  # RGB

            # Clean up
            Path(f.name).unlink()

    def test_load_image_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/to/image.png")

    def test_load_image_accepts_pathlib_path(self):
        """Should accept pathlib.Path objects."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_array = np.array([[100, 150], [200, 50]], dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f.name)

            # Load using Path object
            result = load_image(Path(f.name))
            assert isinstance(result, np.ndarray)

            # Clean up
            Path(f.name).unlink()


from preprocessing.volume import find_best_slice, extract_slice, get_bounding_box


class TestFindBestSlice:
    """Tests for find_best_slice function."""

    def test_find_best_slice_returns_largest_tumor(self):
        """Should return slice index with maximum tumor area."""
        # Create a 3D volume (H, W, Z) with tumor labels
        # Slice 0: 4 tumor pixels, Slice 1: 10 tumor pixels, Slice 2: 2 tumor pixels
        volume = np.zeros((10, 10, 3), dtype=np.int16)
        volume[0:2, 0:2, 0] = 1  # 4 pixels in slice 0
        volume[0:5, 0:2, 1] = 2  # 10 pixels in slice 1
        volume[0:1, 0:2, 2] = 4  # 2 pixels in slice 2

        best_idx, max_area = find_best_slice(volume)

        assert best_idx == 1
        assert max_area == 10

    def test_find_best_slice_custom_labels(self):
        """Should respect custom tumor_labels parameter."""
        volume = np.zeros((10, 10, 2), dtype=np.int16)
        volume[0:5, 0:5, 0] = 3  # 25 pixels, label=3
        volume[0:2, 0:2, 1] = 1  # 4 pixels, label=1

        # Only consider label=1 as tumor
        best_idx, max_area = find_best_slice(volume, tumor_labels=[1])

        assert best_idx == 1
        assert max_area == 4

    def test_find_best_slice_no_tumor(self):
        """Should return index 0 with area 0 when no tumor present."""
        volume = np.zeros((10, 10, 5), dtype=np.int16)

        best_idx, max_area = find_best_slice(volume)

        assert best_idx == 0
        assert max_area == 0


class TestExtractSlice:
    """Tests for extract_slice function."""

    def test_extract_slice_returns_correct_slice(self):
        """Should return the correct 2D slice from 3D volume."""
        volume = np.zeros((10, 10, 5), dtype=np.float32)
        volume[:, :, 2] = 42.0  # Set slice 2 to known value

        result = extract_slice(volume, 2)

        assert result.shape == (10, 10)
        assert np.all(result == 42.0)

    def test_extract_slice_preserves_dtype(self):
        """Should preserve the input array dtype."""
        volume = np.ones((5, 5, 3), dtype=np.float64)
        result = extract_slice(volume, 1)

        assert result.dtype == np.float64


class TestGetBoundingBox:
    """Tests for get_bounding_box function."""

    def test_get_bounding_box_with_valid_mask(self):
        """Should return [x_min, y_min, x_max, y_max] for valid mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1  # rows 2-4, cols 3-6

        bbox = get_bounding_box(mask)

        assert bbox == [3, 2, 6, 4]  # [x_min, y_min, x_max, y_max]

    def test_get_bounding_box_with_empty_mask(self):
        """Should return None for empty mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)

        bbox = get_bounding_box(mask)

        assert bbox is None

    def test_get_bounding_box_single_pixel(self):
        """Should handle single-pixel mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 7] = 1  # Single pixel at row=5, col=7

        bbox = get_bounding_box(mask)

        assert bbox == [7, 5, 7, 5]  # x_min=x_max=7, y_min=y_max=5
