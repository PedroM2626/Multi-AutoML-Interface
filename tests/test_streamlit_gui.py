from streamlit.testing.v1 import AppTest

def test_app_initialization():
    """Verify that app.py compiles and renders the main screen/sidebar without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    
    # Assert no exceptions happened during execution
    assert not at.exception
    
    # Verify key structural elements are present on the rendered page
    assert len(at.sidebar) > 0
