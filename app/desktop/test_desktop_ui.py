import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.desktop.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_builds_and_exposes_top_k_options(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "refresh_api_status", lambda self: None)
    window = MainWindow()
    assert window.windowTitle().startswith("VisionMatch AI")
    assert [window.top_k_combo.itemText(i) for i in range(window.top_k_combo.count())] == [
        "5", "10", "20", "50"
    ]
    window.close()


def test_mode_switch_updates_input_stack(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "refresh_api_status", lambda self: None)
    window = MainWindow()
    window.switch_mode("image")
    assert window.current_mode == "image"
    assert window.input_stack.currentIndex() == 1
    window.switch_mode("text")
    assert window.input_stack.currentIndex() == 0
    window.close()



def test_search_history_is_session_scoped_and_clearable(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "refresh_api_status", lambda self: None)
    window = MainWindow()
    window.search_history.append({
        "mode": "text_to_image",
        "query": "a red car",
        "count": 5,
        "elapsed_ms": 12.5,
        "model": "baseline",
    })
    assert len(window.search_history) == 1
    window.search_history.clear()
    assert window.search_history == []
    window.close()


def test_api_health_status_distinguishes_ready_and_degraded(qt_app, monkeypatch):
    monkeypatch.setattr(MainWindow, "refresh_api_status", lambda self: None)
    window = MainWindow()
    window._on_health_success({"model_loaded": True, "model_type": "gru"})
    assert "Ready" in window.status_label.text()
    window._on_health_success({
        "model_loaded": False,
        "startup_error": "Gallery fingerprint mismatch",
    })
    assert "model unavailable" in window.status_label.text()
    assert "fingerprint mismatch" in window.status_label.toolTip().lower()
    window._on_health_failure("connection refused")
    assert "offline" in window.status_label.text()
    window.close()
