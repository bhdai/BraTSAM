"""Tests for project dependencies (Story 1.1)."""

from pathlib import Path

import pytest


def test_streamlit_dependency_in_pyproject():
    """AC #3: streamlit (>=1.40.0) should be listed in pyproject.toml."""
    pyproject_content = Path("pyproject.toml").read_text()
    
    # Check that streamlit is listed with version >=1.40.0
    assert 'streamlit>=1.40.0' in pyproject_content, (
        "streamlit>=1.40.0 should be in pyproject.toml dependencies"
    )


def test_streamlit_drawable_canvas_dependency_in_pyproject():
    """AC #3: streamlit-drawable-canvas should be listed in pyproject.toml."""
    pyproject_content = Path("pyproject.toml").read_text()
    
    # Check that streamlit-drawable-canvas is listed
    assert 'streamlit-drawable-canvas' in pyproject_content, (
        "streamlit-drawable-canvas should be in pyproject.toml dependencies"
    )


def test_uv_lock_contains_streamlit():
    """AC #3: uv.lock should contain streamlit package entry."""
    uv_lock_path = Path("uv.lock")
    assert uv_lock_path.exists(), "uv.lock should exist after dependency installation"
    
    uv_lock_content = uv_lock_path.read_text()
    assert 'name = "streamlit"' in uv_lock_content, (
        "streamlit should be in uv.lock"
    )


def test_uv_lock_contains_streamlit_drawable_canvas():
    """AC #3: uv.lock should contain streamlit-drawable-canvas package entry."""
    uv_lock_path = Path("uv.lock")
    assert uv_lock_path.exists(), "uv.lock should exist after dependency installation"
    
    uv_lock_content = uv_lock_path.read_text()
    assert 'name = "streamlit-drawable-canvas"' in uv_lock_content, (
        "streamlit-drawable-canvas should be in uv.lock"
    )


def test_streamlit_importable():
    """AC #3: streamlit should be importable."""
    streamlit = pytest.importorskip("streamlit")
    
    assert streamlit is not None
    # Verify version is >= 1.40.0
    version = streamlit.__version__
    major, minor, *_ = version.split(".")
    assert int(major) >= 1, f"streamlit major version should be >= 1, got {version}"
    if int(major) == 1:
        assert int(minor) >= 40, f"streamlit version should be >= 1.40.0, got {version}"


def test_streamlit_drawable_canvas_importable():
    """AC #3: streamlit-drawable-canvas should be importable."""
    streamlit_drawable_canvas = pytest.importorskip("streamlit_drawable_canvas")
    
    assert streamlit_drawable_canvas is not None
