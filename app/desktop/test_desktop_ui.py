import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.desktop.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_builds_and_exposes_top_k_options(qt_app):
    window = MainWindow()
    assert window.windowTitle().startswith("VisionMatch AI")
    assert [window.top_k_combo.itemText(i) for i in range(window.top_k_combo.count())] == [
        "5", "10", "20", "50"
    ]
    window.close()


def test_mode_switch_updates_input_stack(qt_app):
    window = MainWindow()
    window.switch_mode("image")
    assert window.current_mode == "image"
    assert window.input_stack.currentIndex() == 1
    window.switch_mode("text")
    assert window.input_stack.currentIndex() == 0
    window.close()
