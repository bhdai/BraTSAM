"""Tests for Streamlit application entry point (Story 1.1)."""

from pathlib import Path


def test_webapp_app_py_exists():
    """AC #2: webapp/app.py file should exist."""
    assert Path("webapp/app.py").is_file()


def test_webapp_app_py_has_page_title():
    """AC #2: webapp/app.py should contain page title 'BraTSAM'."""
    app_content = Path("webapp/app.py").read_text()
    
    assert 'page_title="BraTSAM"' in app_content, (
        "webapp/app.py should set page_title to 'BraTSAM'"
    )


def test_webapp_app_py_has_set_page_config():
    """AC #2: webapp/app.py should use st.set_page_config()."""
    app_content = Path("webapp/app.py").read_text()
    
    assert "st.set_page_config" in app_content, (
        "webapp/app.py should call st.set_page_config()"
    )


def test_webapp_app_py_has_wide_layout():
    """webapp/app.py should configure wide layout."""
    app_content = Path("webapp/app.py").read_text()
    
    assert 'layout="wide"' in app_content, (
        "webapp/app.py should use layout='wide'"
    )


def test_webapp_app_py_has_logging():
    """webapp/app.py should configure logging."""
    app_content = Path("webapp/app.py").read_text()
    
    assert "import logging" in app_content, (
        "webapp/app.py should import logging module"
    )
    assert "logging.basicConfig" in app_content, (
        "webapp/app.py should configure logging with basicConfig"
    )


def test_webapp_app_py_imports_streamlit():
    """webapp/app.py should import streamlit."""
    app_content = Path("webapp/app.py").read_text()
    
    assert "import streamlit as st" in app_content, (
        "webapp/app.py should import streamlit as st"
    )


def test_webapp_app_py_has_main_function():
    """webapp/app.py should have a main() function."""
    app_content = Path("webapp/app.py").read_text()
    
    assert "def main()" in app_content, (
        "webapp/app.py should define a main() function"
    )


def test_webapp_app_py_has_main_guard():
    """webapp/app.py should have if __name__ == '__main__' guard."""
    app_content = Path("webapp/app.py").read_text()
    
    assert 'if __name__ == "__main__"' in app_content, (
        "webapp/app.py should have main guard"
    )


def test_webapp_app_py_has_page_icon():
    """webapp/app.py should configure brain emoji as page icon."""
    app_content = Path("webapp/app.py").read_text()
    
    assert 'page_icon="🧠"' in app_content, (
        "webapp/app.py should use brain emoji as page_icon"
    )


def test_webapp_app_py_has_sidebar_config():
    """webapp/app.py should configure sidebar to be expanded."""
    app_content = Path("webapp/app.py").read_text()
    
    assert 'initial_sidebar_state="expanded"' in app_content, (
        "webapp/app.py should set initial_sidebar_state='expanded'"
    )
