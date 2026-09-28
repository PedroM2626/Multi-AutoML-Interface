from streamlit.testing.v1 import AppTest

def test_app_initialization():
    """Verify that app.py compiles and renders the main screen/sidebar without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    
    # Assert no exceptions happened during execution
    assert not at.exception
    
    # Verify key structural elements are present on the rendered page
    assert len(at.sidebar) > 0


def _patch_server_address(monkeypatch, value):
    import streamlit as st

    real_get_option = st.get_option

    def fake_get_option(key):
        return value if key == "server.address" else real_get_option(key)

    monkeypatch.setattr(st, "get_option", fake_get_option)


def _has_dagshub_input(at):
    return any("DagsHub" in (c.label or "") for c in at.checkbox)


def test_dagshub_token_panel_needs_loopback_binding(monkeypatch):
    _patch_server_address(monkeypatch, "127.0.0.1")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert not at.exception
    assert _has_dagshub_input(at)


def test_dagshub_token_panel_hidden_when_bound_publicly(monkeypatch):
    _patch_server_address(monkeypatch, "0.0.0.0")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert not at.exception
    assert not _has_dagshub_input(at)
    assert any(
        "DagsHub" in (m.value or "") for m in at.info
    ), "the disabled panel should say why it is disabled"
