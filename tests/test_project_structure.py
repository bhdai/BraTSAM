"""Tests for project directory structure (Story 1.1)."""

from pathlib import Path


def test_webapp_directory_exists():
    """AC #1: webapp/ directory should exist."""
    assert Path("webapp").is_dir()


def test_webapp_components_directory_exists():
    """AC #1: webapp/components/ directory should exist."""
    assert Path("webapp/components").is_dir()


def test_webapp_utils_directory_exists():
    """AC #1: webapp/utils/ directory should exist."""
    assert Path("webapp/utils").is_dir()


def test_preprocessing_directory_exists():
    """AC #1: preprocessing/ directory should exist."""
    assert Path("preprocessing").is_dir()


def test_tests_directory_exists():
    """AC #1: tests/ directory should exist."""
    assert Path("tests").is_dir()


def test_webapp_init_exists():
    """AC #5: webapp/__init__.py should exist for package import."""
    assert Path("webapp/__init__.py").is_file()


def test_webapp_components_init_exists():
    """AC #5: webapp/components/__init__.py should exist for package import."""
    assert Path("webapp/components/__init__.py").is_file()


def test_webapp_utils_init_exists():
    """AC #5: webapp/utils/__init__.py should exist for package import."""
    assert Path("webapp/utils/__init__.py").is_file()


def test_preprocessing_init_exists():
    """AC #5: preprocessing/__init__.py should exist for package import."""
    assert Path("preprocessing/__init__.py").is_file()


def test_tests_init_exists():
    """AC #5: tests/__init__.py should exist for package import."""
    assert Path("tests/__init__.py").is_file()


def test_packages_importable():
    """AC #5: All new packages should be importable."""
    import webapp
    import webapp.components
    import webapp.utils
    import preprocessing
    import tests
    
    # Verify imports succeeded
    assert webapp is not None
    assert webapp.components is not None
    assert webapp.utils is not None
    assert preprocessing is not None
    assert tests is not None
