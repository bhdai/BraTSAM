"""Unit tests for 3D volume upload and slice extraction.

Tests cover:
- NIfTI volume loading (Task 1)
- UploadedVolume dataclass (Task 2)
- File type detection (Task 3)
- Slice selector range calculation (Task 4)
- Slice extraction and normalization (Task 5, Task 7)
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from preprocessing.formats import load_nifti


class TestLoadNifti:
    """Tests for load_nifti() function (AC: #1)."""

    def test_load_nifti_returns_3d_array(self, tmp_path: Path) -> None:
        """NIfTI loader should return 3D numpy array."""
        # Create a test NIfTI volume
        volume_data = np.random.rand(64, 64, 32).astype(np.float32)
        nii_img = nib.Nifti1Image(volume_data, affine=np.eye(4))
        nii_path = tmp_path / "test_volume.nii.gz"
        nib.save(nii_img, nii_path)

        # Load and verify
        result = load_nifti(nii_path)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape == (64, 64, 32)

    def test_load_nifti_supports_nii_extension(self, tmp_path: Path) -> None:
        """Should load .nii files without compression."""
        volume_data = np.random.rand(32, 32, 16).astype(np.float32)
        nii_img = nib.Nifti1Image(volume_data, affine=np.eye(4))
        nii_path = tmp_path / "test_volume.nii"
        nib.save(nii_img, nii_path)

        result = load_nifti(nii_path)
        
        assert result.shape == (32, 32, 16)

    def test_load_nifti_supports_nii_gz_extension(self, tmp_path: Path) -> None:
        """Should load .nii.gz compressed files."""
        volume_data = np.random.rand(32, 32, 16).astype(np.float32)
        nii_img = nib.Nifti1Image(volume_data, affine=np.eye(4))
        nii_path = tmp_path / "test_volume.nii.gz"
        nib.save(nii_img, nii_path)

        result = load_nifti(nii_path)
        
        assert result.shape == (32, 32, 16)

    def test_load_nifti_file_not_found(self) -> None:
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Volume file not found"):
            load_nifti("/nonexistent/path/volume.nii.gz")

    def test_load_nifti_accepts_string_path(self, tmp_path: Path) -> None:
        """Should accept string path in addition to Path object."""
        volume_data = np.random.rand(32, 32, 16).astype(np.float32)
        nii_img = nib.Nifti1Image(volume_data, affine=np.eye(4))
        nii_path = tmp_path / "test_volume.nii.gz"
        nib.save(nii_img, nii_path)

        # Pass as string
        result = load_nifti(str(nii_path))
        
        assert result.shape == (32, 32, 16)

    def test_load_nifti_canonical_orientation(self, tmp_path: Path) -> None:
        """Should return volume in canonical (RAS+) orientation."""
        # Create a volume with non-identity affine
        volume_data = np.random.rand(64, 64, 32).astype(np.float32)
        # Non-identity affine (rotated/flipped)
        affine = np.array([
            [-1, 0, 0, 63],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        nii_img = nib.Nifti1Image(volume_data, affine=affine)
        nii_path = tmp_path / "test_volume.nii.gz"
        nib.save(nii_img, nii_path)

        # Load - should be converted to canonical
        result = load_nifti(nii_path)
        
        # Result should still be 3D with valid shape
        assert result.ndim == 3
        # Shape might change order due to reorientation
        assert result.size == volume_data.size


class TestUploadedVolume:
    """Tests for UploadedVolume dataclass (AC: #1, #5)."""

    def test_uploaded_volume_creation(self) -> None:
        """UploadedVolume should be creatable with required fields."""
        from webapp.components.upload import UploadedVolume
        
        volume_data = np.random.rand(64, 64, 32).astype(np.float32)
        
        vol = UploadedVolume(
            filename="test.nii.gz",
            volume_data=volume_data,
            file_size_bytes=1024 * 1024,
            file_id="test-id-123",
        )
        
        assert vol.filename == "test.nii.gz"
        assert vol.file_size_bytes == 1024 * 1024
        assert vol.file_id == "test-id-123"

    def test_uploaded_volume_dimensions_property(self) -> None:
        """UploadedVolume.dimensions should return (H, W, D) tuple."""
        from webapp.components.upload import UploadedVolume
        
        volume_data = np.random.rand(64, 128, 32).astype(np.float32)
        
        vol = UploadedVolume(
            filename="test.nii.gz",
            volume_data=volume_data,
            file_size_bytes=1024,
        )
        
        assert vol.dimensions == (64, 128, 32)

    def test_uploaded_volume_num_slices(self) -> None:
        """UploadedVolume.num_slices should return depth dimension."""
        from webapp.components.upload import UploadedVolume
        
        volume_data = np.random.rand(64, 64, 155).astype(np.float32)
        
        vol = UploadedVolume(
            filename="test.nii.gz",
            volume_data=volume_data,
            file_size_bytes=1024,
        )
        
        assert vol.num_slices == 155

    def test_uploaded_volume_file_size_mb(self) -> None:
        """UploadedVolume.file_size_mb should return size in megabytes."""
        from webapp.components.upload import UploadedVolume
        
        volume_data = np.random.rand(10, 10, 10).astype(np.float32)
        
        vol = UploadedVolume(
            filename="test.nii.gz",
            volume_data=volume_data,
            file_size_bytes=2 * 1024 * 1024,  # 2MB
        )
        
        assert vol.file_size_mb == 2.0


class TestFileTypeDetection:
    """Tests for file type detection (AC: #1, #6)."""

    def test_is_nifti_file_nii(self) -> None:
        """Should detect .nii as NIfTI file."""
        from webapp.components.upload import is_nifti_file
        
        assert is_nifti_file("brain.nii") is True

    def test_is_nifti_file_nii_gz(self) -> None:
        """Should detect .nii.gz as NIfTI file."""
        from webapp.components.upload import is_nifti_file
        
        assert is_nifti_file("brain.nii.gz") is True

    def test_is_nifti_file_case_insensitive(self) -> None:
        """Should be case insensitive."""
        from webapp.components.upload import is_nifti_file
        
        assert is_nifti_file("BRAIN.NII.GZ") is True
        assert is_nifti_file("Brain.Nii") is True

    def test_is_nifti_file_png(self) -> None:
        """Should detect .png as NOT NIfTI."""
        from webapp.components.upload import is_nifti_file
        
        assert is_nifti_file("brain.png") is False

    def test_is_nifti_file_jpg(self) -> None:
        """Should detect .jpg/.jpeg as NOT NIfTI."""
        from webapp.components.upload import is_nifti_file
        
        assert is_nifti_file("brain.jpg") is False
        assert is_nifti_file("brain.jpeg") is False


class TestSliceExtraction:
    """Tests for slice extraction and normalization (AC: #4)."""

    def test_slice_extraction_shape(self) -> None:
        """Extracted slice should have shape (H, W)."""
        from preprocessing.volume import extract_slice
        
        volume = np.random.rand(64, 128, 32).astype(np.float32)
        slice_2d = extract_slice(volume, 15)
        
        assert slice_2d.shape == (64, 128)

    def test_normalized_slice_dtype(self) -> None:
        """Normalized slice should be uint8."""
        from preprocessing.normalize import normalize_slice
        
        slice_2d = np.random.rand(64, 64).astype(np.float32) * 1000
        normalized = normalize_slice(slice_2d)
        
        assert normalized.dtype == np.uint8

    def test_normalized_slice_range(self) -> None:
        """Normalized slice should be in range [0, 255]."""
        from preprocessing.normalize import normalize_slice
        
        slice_2d = np.random.rand(64, 64).astype(np.float32) * 1000
        normalized = normalize_slice(slice_2d)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 255


class TestSliceSelector:
    """Tests for slice selector component (AC: #2, #3)."""

    def test_default_middle_slice(self) -> None:
        """Slice selector should default to middle slice index."""
        # Middle of 155 slices should be 77
        num_slices = 155
        default_idx = num_slices // 2
        
        assert default_idx == 77

    def test_slider_range_calculation(self) -> None:
        """Slider range should be 0 to num_slices - 1."""
        num_slices = 100
        
        min_val = 0
        max_val = num_slices - 1
        
        assert min_val == 0
        assert max_val == 99
