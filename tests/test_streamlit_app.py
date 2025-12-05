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


# Story 2.4 Tests: Confidence Display Integration


def test_webapp_app_imports_confidence_indicator():
    """AC #3, #5: webapp/app.py should import confidence indicator."""
    app_content = Path("webapp/app.py").read_text()

    assert "from webapp.components.confidence_indicator import" in app_content, (
        "webapp/app.py should import from confidence_indicator"
    )
    assert "render_confidence_indicator" in app_content, (
        "webapp/app.py should import render_confidence_indicator"
    )


def test_webapp_app_imports_pipeline_result():
    """AC #5: webapp/app.py should import PipelineResult."""
    app_content = Path("webapp/app.py").read_text()

    assert "from webapp.utils.inference import PipelineResult" in app_content, (
        "webapp/app.py should import PipelineResult for inference results"
    )


def test_webapp_app_has_display_inference_results():
    """AC #3, #5: webapp/app.py should have display_inference_results function."""
    app_content = Path("webapp/app.py").read_text()

    assert "def display_inference_results" in app_content, (
        "webapp/app.py should define display_inference_results function"
    )


def test_webapp_app_display_function_uses_confidence_indicator():
    """AC #3: display_inference_results should use render_confidence_indicator."""
    app_content = Path("webapp/app.py").read_text()

    # Check that the display function calls render_confidence_indicator
    assert "render_confidence_indicator(result)" in app_content, (
        "display_inference_results should call render_confidence_indicator"
    )


def test_webapp_app_shows_threshold_explanation():
    """AC #5: App should show tooltip explaining classification thresholds."""
    app_content = Path("webapp/app.py").read_text()

    assert "Thresholds" in app_content, (
        "webapp/app.py should display threshold explanation"
    )
    assert "90%" in app_content, (
        "webapp/app.py should mention 90% threshold for auto-approved"
    )
    assert "50" in app_content, (
        "webapp/app.py should mention 50% threshold for needs-review"
    )


# Story 3.1 Tests: Interactive Image Viewer Component Integration


def test_webapp_app_imports_render_image_viewer():
    """Story 3.1 AC #5: webapp/app.py should import render_image_viewer."""
    app_content = Path("webapp/app.py").read_text()

    assert "from webapp.components.viewer import render_image_viewer" in app_content, (
        "webapp/app.py should import render_image_viewer from viewer component"
    )


def test_webapp_app_uses_render_image_viewer():
    """Story 3.1 AC #5: webapp/app.py should use viewer component for display.
    
    Updated for Story 3.3: Now checks for render_interactive_viewer or render_image_viewer.
    """
    app_content = Path("webapp/app.py").read_text()

    assert "render_interactive_viewer(" in app_content or "render_image_viewer(" in app_content, (
        "webapp/app.py should use viewer function (render_image_viewer or render_interactive_viewer) for image display"
    )


def test_webapp_app_has_queue_master_layout():
    """Story 3.1 AC #5: webapp/app.py should implement Queue Master layout columns."""
    app_content = Path("webapp/app.py").read_text()

    assert "st.columns([1, 3])" in app_content, (
        "webapp/app.py should use st.columns([1, 3]) for Queue Master layout"
    )


def test_webapp_app_has_review_queue_placeholder():
    """Story 3.1 AC #5: webapp/app.py should have placeholder for review queue."""
    app_content = Path("webapp/app.py").read_text()

    assert "Review Queue" in app_content, (
        "webapp/app.py should have Review Queue placeholder"
    )
