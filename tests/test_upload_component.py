"""Unit tests for the upload component.

Tests cover:
- File type validation (PNG, JPG, JPEG)
- File size validation (under 10MB)
- Session state storage for uploaded images
- Image replacement behavior
- Error message display for invalid formats
- Zombie state clearing on validation failure
- File caching to avoid redundant processing
"""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import UnidentifiedImageError

from webapp.components.upload import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    UploadedImage,
    clear_upload,
    render_upload_component,
)


class TestUploadedImageDataclass:
    """Tests for the UploadedImage dataclass."""

    def test_uploaded_image_creation(self) -> None:
        """Test that UploadedImage can be created with valid data."""
        test_array = np.zeros((100, 200, 3), dtype=np.uint8)
        image = UploadedImage(
            filename="test.png",
            data=test_array,
            file_size_bytes=1024,
        )
        
        assert image.filename == "test.png"
        assert image.data.shape == (100, 200, 3)
        assert image.file_size_bytes == 1024
        assert isinstance(image.upload_time, datetime)

    def test_file_size_mb_property(self) -> None:
        """Test file_size_mb property correctly converts bytes to MB."""
        test_array = np.zeros((10, 10, 3), dtype=np.uint8)
        image = UploadedImage(
            filename="test.png",
            data=test_array,
            file_size_bytes=5 * 1024 * 1024,  # 5MB in bytes
        )
        
        assert image.file_size_mb == pytest.approx(5.0)

    def test_dimensions_property(self) -> None:
        """Test dimensions property returns (height, width) tuple."""
        test_array = np.zeros((480, 640, 3), dtype=np.uint8)  # height=480, width=640
        image = UploadedImage(
            filename="test.png",
            data=test_array,
            file_size_bytes=1024,
        )
        
        assert image.dimensions == (480, 640)


class TestFileSizeConstants:
    """Tests for file size constants."""

    def test_max_file_size_mb_is_10(self) -> None:
        """Test that MAX_FILE_SIZE_MB is set to 10."""
        assert MAX_FILE_SIZE_MB == 10

    def test_max_file_size_bytes_calculation(self) -> None:
        """Test that MAX_FILE_SIZE_BYTES is correctly calculated."""
        assert MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024


class TestRenderUploadComponent:
    """Tests for render_upload_component function."""

    @patch("webapp.components.upload.st")
    def test_file_uploader_configured_with_correct_types(
        self, mock_st: MagicMock
    ) -> None:
        """Test that st.file_uploader is called with correct file types."""
        mock_st.file_uploader.return_value = None
        mock_st.session_state = {}
        
        render_upload_component()
        
        mock_st.file_uploader.assert_called_once()
        call_kwargs = mock_st.file_uploader.call_args
        assert "type" in call_kwargs.kwargs
        assert set(call_kwargs.kwargs["type"]) == {"png", "jpg", "jpeg"}

    @patch("webapp.components.upload.st")
    def test_returns_none_when_no_file_uploaded(self, mock_st: MagicMock) -> None:
        """Test that None is returned when no file is uploaded."""
        mock_st.file_uploader.return_value = None
        mock_st.session_state = {}
        
        result = render_upload_component()
        
        assert result is None

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_file_size_validation_rejects_large_files(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that files over 10MB are rejected with error message."""
        mock_file = MagicMock()
        mock_file.size = 11 * 1024 * 1024  # 11MB - over limit
        mock_file.name = "large_file.png"
        mock_file.file_id = "test-file-id-large"
        mock_st.file_uploader.return_value = mock_file
        mock_st.session_state = {}
        
        result = render_upload_component()
        
        mock_st.error.assert_called()
        error_message = mock_st.error.call_args[0][0]
        assert "too large" in error_message.lower() or "File too large" in error_message
        assert result is None

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_file_size_validation_accepts_files_under_limit(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that files under 10MB are accepted."""
        mock_file = MagicMock()
        mock_file.size = 5 * 1024 * 1024  # 5MB - under limit
        mock_file.name = "valid_file.png"
        mock_file.file_id = "test-file-id-valid"
        mock_st.file_uploader.return_value = mock_file
        mock_st.session_state = {}
        
        # Mock PIL Image
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_image.open.return_value = mock_pil_image
        
        # Mock numpy array conversion
        with patch("webapp.components.upload.np.array") as mock_np_array:
            mock_np_array.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            result = render_upload_component()
        
        # Should not show error for valid file size
        assert not any(
            "too large" in str(call).lower() 
            for call in mock_st.error.call_args_list
        ) if mock_st.error.called else True
        assert result is not None

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_session_state_storage_for_uploaded_image(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that uploaded image is stored in session state."""
        mock_file = MagicMock()
        mock_file.size = 1024
        mock_file.name = "test.png"
        mock_file.file_id = "test-file-id-storage"
        mock_st.file_uploader.return_value = mock_file
        mock_st.session_state = {}
        
        # Mock PIL Image
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_image.open.return_value = mock_pil_image
        
        with patch("webapp.components.upload.np.array") as mock_np_array:
            mock_np_array.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            render_upload_component()
        
        assert "uploaded_image" in mock_st.session_state
        assert isinstance(mock_st.session_state["uploaded_image"], UploadedImage)

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_image_replacement_when_new_file_uploaded(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that new upload replaces previous image in session state."""
        # Set up existing image in session state
        old_image = UploadedImage(
            filename="old.png",
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            file_size_bytes=512,
            file_id="old-file-id",
        )
        mock_st.session_state = {"uploaded_image": old_image}
        
        # Upload new file with different file_id
        mock_file = MagicMock()
        mock_file.size = 2048
        mock_file.name = "new.png"
        mock_file.file_id = "new-file-id"
        mock_st.file_uploader.return_value = mock_file
        
        # Mock PIL Image
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_image.open.return_value = mock_pil_image
        
        with patch("webapp.components.upload.np.array") as mock_np_array:
            mock_np_array.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
            
            render_upload_component()
        
        # Verify new image replaced old
        assert mock_st.session_state["uploaded_image"].filename == "new.png"
        assert mock_st.session_state["uploaded_image"].file_size_bytes == 2048

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_error_handling_for_invalid_image_data(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that errors during image loading are handled gracefully."""
        mock_file = MagicMock()
        mock_file.size = 1024
        mock_file.name = "corrupted.png"
        mock_file.file_id = "test-file-id-corrupted"
        mock_st.file_uploader.return_value = mock_file
        mock_st.session_state = {}
        
        # Make Image.open raise an exception
        mock_image.open.side_effect = Exception("Corrupted image data")
        
        result = render_upload_component()
        
        mock_st.error.assert_called()
        assert result is None


class TestZombieStateClearing:
    """Tests for zombie state clearing functionality."""

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_clears_session_state_when_file_too_large(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that session state is cleared when file size validation fails."""
        # Set up existing image in session state
        old_image = UploadedImage(
            filename="old.png",
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            file_size_bytes=512,
            file_id="old-file-id",
        )
        mock_st.session_state = {"uploaded_image": old_image}
        
        # Upload file that's too large
        mock_file = MagicMock()
        mock_file.size = 11 * 1024 * 1024  # 11MB - over limit
        mock_file.name = "large_file.png"
        mock_file.file_id = "large-file-id"
        mock_st.file_uploader.return_value = mock_file
        
        render_upload_component()
        
        # Session state should be cleared (zombie state fix)
        assert mock_st.session_state["uploaded_image"] is None

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_clears_session_state_when_image_load_fails(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that session state is cleared when image loading fails."""
        # Set up existing image in session state
        old_image = UploadedImage(
            filename="old.png",
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            file_size_bytes=512,
            file_id="old-file-id",
        )
        mock_st.session_state = {"uploaded_image": old_image}
        
        # Upload corrupted file
        mock_file = MagicMock()
        mock_file.size = 1024
        mock_file.name = "corrupted.png"
        mock_file.file_id = "corrupted-file-id"
        mock_st.file_uploader.return_value = mock_file
        
        mock_image.open.side_effect = Exception("Corrupted image data")
        
        render_upload_component()
        
        # Session state should be cleared (zombie state fix)
        assert mock_st.session_state["uploaded_image"] is None

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_clears_session_state_on_unidentified_image_error(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that session state is cleared when UnidentifiedImageError occurs."""
        # Set up existing image in session state
        old_image = UploadedImage(
            filename="old.png",
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            file_size_bytes=512,
            file_id="old-file-id",
        )
        mock_st.session_state = {"uploaded_image": old_image}
        
        # Upload invalid file
        mock_file = MagicMock()
        mock_file.size = 1024
        mock_file.name = "not_an_image.png"
        mock_file.file_id = "invalid-file-id"
        mock_st.file_uploader.return_value = mock_file
        
        mock_image.open.side_effect = UnidentifiedImageError("Not a valid image")
        
        result = render_upload_component()
        
        # Session state should be cleared (zombie state fix)
        assert mock_st.session_state["uploaded_image"] is None
        # User-friendly error message should be shown
        mock_st.error.assert_called()
        error_message = mock_st.error.call_args[0][0]
        assert "Invalid image file" in error_message
        assert result is None

    @patch("webapp.components.upload.clear_upload")
    @patch("webapp.components.upload.st")
    def test_clears_session_state_when_no_file_uploaded(
        self, mock_st: MagicMock, mock_clear_upload: MagicMock
    ) -> None:
        """Test that session state is cleared when user removes file."""
        # Set up existing image in session state
        old_image = UploadedImage(
            filename="old.png",
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            file_size_bytes=512,
            file_id="old-file-id",
        )
        mock_st.session_state = {"uploaded_image": old_image}
        mock_st.file_uploader.return_value = None
        
        render_upload_component()
        
        # clear_upload should be called when file is removed
        mock_clear_upload.assert_called_once()


class TestFileCaching:
    """Tests for file caching to avoid redundant processing."""

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_returns_cached_image_for_same_file_id(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that cached image is returned when file_id matches."""
        # Set up existing image in session state with specific file_id
        existing_image = UploadedImage(
            filename="cached.png",
            data=np.zeros((100, 100, 3), dtype=np.uint8),
            file_size_bytes=1024,
            file_id="same-file-id",
        )
        mock_st.session_state = {"uploaded_image": existing_image}
        
        # Upload file with same file_id
        mock_file = MagicMock()
        mock_file.size = 1024
        mock_file.name = "cached.png"
        mock_file.file_id = "same-file-id"
        mock_st.file_uploader.return_value = mock_file
        
        result = render_upload_component()
        
        # Should return cached image without calling Image.open
        mock_image.open.assert_not_called()
        assert result is existing_image

    @patch("webapp.components.upload.Image")
    @patch("webapp.components.upload.st")
    def test_processes_new_file_when_file_id_differs(
        self, mock_st: MagicMock, mock_image: MagicMock
    ) -> None:
        """Test that new file is processed when file_id is different."""
        # Set up existing image in session state
        existing_image = UploadedImage(
            filename="old.png",
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            file_size_bytes=512,
            file_id="old-file-id",
        )
        mock_st.session_state = {"uploaded_image": existing_image}
        
        # Upload file with different file_id
        mock_file = MagicMock()
        mock_file.size = 2048
        mock_file.name = "new.png"
        mock_file.file_id = "new-file-id"
        mock_st.file_uploader.return_value = mock_file
        
        # Mock PIL Image
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_image.open.return_value = mock_pil_image
        
        with patch("webapp.components.upload.np.array") as mock_np_array:
            mock_np_array.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
            
            result = render_upload_component()
        
        # Should process new file
        mock_image.open.assert_called_once()
        assert result.filename == "new.png"
        assert result.file_id == "new-file-id"


class TestClearUpload:
    """Tests for clear_upload helper function."""

    @patch("webapp.components.upload.st")
    def test_clear_upload_removes_image_from_session_state(
        self, mock_st: MagicMock
    ) -> None:
        """Test that clear_upload sets session state to None."""
        mock_st.session_state = {
            "uploaded_image": UploadedImage(
                filename="test.png",
                data=np.zeros((10, 10, 3), dtype=np.uint8),
                file_size_bytes=100,
            )
        }
        
        clear_upload()
        
        assert mock_st.session_state["uploaded_image"] is None

    @patch("webapp.components.upload.st")
    def test_clear_upload_handles_missing_key(self, mock_st: MagicMock) -> None:
        """Test that clear_upload handles case where key doesn't exist."""
        mock_st.session_state = {}
        
        # Should not raise an error
        clear_upload()
        
        assert mock_st.session_state.get("uploaded_image") is None
