
from pathlib import Path
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QFileDialog,
    QStackedWidget,
    QMessageBox,
    QSizePolicy,
    QComboBox,
)


COLORS = {
    "background": "#F8F9FC",
    "surface": "#FFFFFF",
    "primary": "#6258E8",
    "primary_dark": "#5146D8",
    "primary_light": "#F0EFFF",
    "text": "#171923",
    "muted": "#85899A",
    "border": "#E8EAF1",
}


class RetrievalWorker(QThread):
    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, api_base, mode, query="", image_path=None, top_k=5, model="baseline"):
        super().__init__()
        self.api_base = api_base.rstrip("/")
        self.mode = mode
        self.query = query
        self.image_path = image_path
        self.top_k = top_k
        self.model = model

    def run(self):
        try:
            if self.mode == "text":
                url = f"{self.api_base}/search/text"
                payload = json.dumps({
                    "query": self.query,
                    "top_k": self.top_k,
                    "model": self.model,
                }).encode("utf-8")
                request = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
            else:
                boundary = "----VisionMatchDesktopBoundary"
                path = Path(self.image_path)
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                with path.open("rb") as image_file:
                    image_bytes = image_file.read()
                parts = [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
                    f"Content-Type: {mime}\r\n\r\n".encode(),
                    image_bytes,
                    f"\r\n--{boundary}--\r\n".encode(),
                ]
                url = f"{self.api_base}/search/image?{urllib.parse.urlencode({'top_k': self.top_k, 'model': self.model})}"
                request = urllib.request.Request(
                    url, data=b"".join(parts),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )

            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.succeeded.emit(payload)
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8")).get("detail", str(error))
            except Exception:
                detail = str(error)
            self.failed.emit(str(detail))
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.selected_image_path = None
        self.current_mode = "text"
        self.api_base = os.getenv("VISIONMATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")
        self.model_type = os.getenv("VISIONMATCH_MODEL_TYPE", "baseline").strip().lower()
        self.search_worker = None

        self.setWindowTitle("VisionMatch AI | Cross-Modal Retrieval")
        self.resize(1440, 900)
        self.setMinimumSize(950, 650)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8F9FC;
            }

            QWidget {
                font-family: Segoe UI, Arial;
                color: #171923;
            }

            QPushButton {
                border: none;
                border-radius: 9px;
                padding: 11px 16px;
                font-size: 13px;
                font-weight: 600;
            }

            QTextEdit {
                background-color: #FBFBFD;
                border: 1px solid #E5E7EF;
                border-radius: 11px;
                padding: 14px;
                font-size: 15px;
                selection-background-color: #6258E8;
            }

            QTextEdit:focus {
                border: 2px solid #8279EE;
            }
        """)

        self.build_ui()

    # -----------------------------------------------------
    # Main Layout
    # -----------------------------------------------------

    def build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)

        self.content = QWidget()
        self.content.setStyleSheet(
            "background-color: #F8F9FC;"
        )

        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        content_layout.addWidget(self.create_header())

        self.pages = QStackedWidget()
        self.pages.addWidget(self.create_home_page())

        content_layout.addWidget(self.pages)

        main_layout.addWidget(self.content, 1)

        self.switch_mode("text")

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    def create_sidebar(self):

        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: white;
                border-right: 1px solid #E9EAF1;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 25, 20, 20)
        layout.setSpacing(12)

        logo = QLabel("🧠  VisionMatch")
        logo.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #6258E8;
            padding: 5px;
        """)

        subtitle = QLabel("       AI RETRIEVAL")
        subtitle.setStyleSheet("""
            color: #8B8FA2;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 2px;
        """)

        layout.addWidget(logo)
        layout.addWidget(subtitle)
        layout.addSpacing(35)

        workspace = QLabel("WORKSPACE")
        workspace.setStyleSheet("""
            color: #A0A3B1;
            font-size: 10px;
            font-weight: bold;
            padding-left: 10px;
        """)

        layout.addWidget(workspace)

        self.overview_btn = self.create_nav_button(
            "Overview", active=True
        )

        self.text_btn = self.create_nav_button("Text to Image")
        self.image_btn = self.create_nav_button("Image to Text")

        self.overview_btn.clicked.connect(
            lambda: self.switch_mode("text")
        )
        self.text_btn.clicked.connect(
            lambda: self.switch_mode("text")
        )
        self.image_btn.clicked.connect(
            lambda: self.switch_mode("image")
        )

        layout.addWidget(self.overview_btn)
        layout.addWidget(self.text_btn)
        layout.addWidget(self.image_btn)

        layout.addSpacing(25)

        general = QLabel("GENERAL")
        general.setStyleSheet("""
            color: #A0A3B1;
            font-size: 10px;
            font-weight: bold;
            padding-left: 10px;
        """)

        layout.addWidget(general)

        history_btn = self.create_nav_button("Search History")
        about_btn = self.create_nav_button("About Project")

        history_btn.clicked.connect(self.show_history)
        about_btn.clicked.connect(self.show_about)

        layout.addWidget(history_btn)
        layout.addWidget(about_btn)

        layout.addStretch()

        info_card = QFrame()
        info_card.setStyleSheet("""
            QFrame {
                background-color: #F3F1FF;
                border: 1px solid #E9E7FF;
                border-radius: 13px;
            }
        """)

        info_layout = QVBoxLayout(info_card)

        info_title = QLabel("✨ Powered by Deep Learning")
        info_title.setWordWrap(True)
        info_title.setStyleSheet(
            "font-weight: bold; font-size: 12px; border: none;"
        )

        info_text = QLabel(
            "Discover semantic connections between "
            "images and natural language."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("""
            color: #777B8C;
            font-size: 11px;
            border: none;
        """)

        info_layout.addWidget(info_title)
        info_layout.addWidget(info_text)
        layout.addWidget(info_card)

        footer = QLabel("VisionMatch AI · Academic Project")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("""
            color: #A0A3B1;
            font-size: 9px;
            padding-top: 12px;
        """)

        layout.addWidget(footer)

        return sidebar

    def create_nav_button(self, text, active=False):

        button = QPushButton(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(42)

        self.set_nav_style(button, active)

        return button

    def set_nav_style(self, button, active):

        if active:
            button.setStyleSheet("""
                QPushButton {
                    background-color: #F0EFFF;
                    color: #6258E8;
                    text-align: left;
                    padding-left: 14px;
                }
            """)
        else:
            button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #777B8C;
                    text-align: left;
                    padding-left: 14px;
                }

                QPushButton:hover {
                    background-color: #F5F5FA;
                }
            """)

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    def create_header(self):

        header = QFrame()
        header.setFixedHeight(76)

        header.setStyleSheet("""
            QFrame {
                background-color: white;
                border-bottom: 1px solid #E9EAF1;
            }
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(30, 10, 30, 10)

        title_layout = QVBoxLayout()

        breadcrumb = QLabel("Workspace / Overview")
        breadcrumb.setStyleSheet(
            "color: #9699A8; font-size: 11px;"
        )

        title = QLabel("AI Retrieval Studio")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        title_layout.addWidget(breadcrumb)
        title_layout.addWidget(title)

        layout.addLayout(title_layout)
        layout.addStretch()

        status = QLabel("●  Backend not connected")
        status.setStyleSheet("""
            color: #A47728;
            background-color: #FFFAF0;
            border: 1px solid #F1DCAE;
            border-radius: 14px;
            padding: 9px 13px;
            font-size: 11px;
            font-weight: bold;
        """)

        layout.addWidget(status)

        return header

    # -----------------------------------------------------
    # Home Page
    # -----------------------------------------------------

    def create_home_page(self):

        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 22, 30, 22)
        layout.setSpacing(14)

        # Hero

        hero = QFrame()
        hero.setMinimumHeight(165)

        hero.setStyleSheet("""
            QFrame {
                background-color: #393276;
                border-radius: 20px;
            }
        """)

        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(30, 20, 30, 20)
        hero_layout.setSpacing(8)

        tag = QLabel("✦  CROSS-MODAL AI TECHNOLOGY")
        tag.setStyleSheet("""
            color: #E5E1FF;
            font-size: 11px;
            font-weight: bold;
        """)

        hero_title = QLabel("Search beyond words and images.")
        hero_title.setWordWrap(True)
        hero_title.setStyleSheet("""
            color: white;
            font-size: 27px;
            font-weight: bold;
        """)

        hero_description = QLabel(
            "Explore the connection between visual content and natural language.\n"
            "Find relevant images from text, or discover descriptions from images."
        )
        hero_description.setWordWrap(True)
        hero_description.setStyleSheet("""
            color: #D5D1F5;
            font-size: 12px;
        """)

        hero_layout.addWidget(tag)
        hero_layout.addWidget(hero_title)
        hero_layout.addWidget(hero_description)

        layout.addWidget(hero)

        # Search section headings

        section_title = QLabel("What are you looking for?")
        section_title.setStyleSheet("""
            font-size: 21px;
            font-weight: bold;
        """)

        section_description = QLabel(
            "Choose how you want to search across the image-text space."
        )
        section_description.setStyleSheet(
            "color: #85899A; font-size: 12px;"
        )

        layout.addWidget(section_title)
        layout.addWidget(section_description)

        # Search Card

        card = QFrame()
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #E8EAF1;
                border-radius: 16px;
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 18, 22, 18)
        card_layout.setSpacing(13)

        # Mode Buttons (kept compact)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)

        self.text_mode_btn = QPushButton("⌕  Text to Image")
        self.image_mode_btn = QPushButton("▧  Image to Text")

        self.text_mode_btn.clicked.connect(
            lambda: self.switch_mode("text")
        )
        self.image_mode_btn.clicked.connect(
            lambda: self.switch_mode("image")
        )

        mode_layout.addWidget(self.text_mode_btn)
        mode_layout.addWidget(self.image_mode_btn)
        mode_layout.addStretch()

        card_layout.addLayout(mode_layout)

        # Input stack

        self.input_stack = QStackedWidget()
        self.input_stack.setMinimumHeight(245)

        self.input_stack.addWidget(self.create_text_input())
        self.input_stack.addWidget(self.create_image_input())

        card_layout.addWidget(self.input_stack, 1)

        # Message

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("""
            color: #5C53C7;
            background-color: #F7F6FF;
            border: 1px solid #E8E5FF;
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
        """)

        self.message_label.hide()
        card_layout.addWidget(self.message_label)

        # Bottom row

        bottom_layout = QHBoxLayout()

        hint = QLabel(
            "✦  Semantic matching powered by deep learning"
        )
        hint.setStyleSheet(
            "color: #999CAC; font-size: 10px;"
        )

        bottom_layout.addWidget(hint)
        bottom_layout.addStretch()

        self.search_btn = QPushButton("⌕  Search  →")
        self.search_btn.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.search_btn.setMinimumHeight(45)

        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #6258E8;
                color: white;
                padding: 12px 28px;
                font-size: 14px;
            }

            QPushButton:hover {
                background-color: #5146D8;
            }
        """)

        self.top_k_combo = QComboBox()
        self.top_k_combo.addItems(["5", "10", "20", "50"])
        self.top_k_combo.setCurrentText("5")
        self.top_k_combo.setMinimumHeight(40)
        self.top_k_combo.setToolTip("Number of retrieval results")
        bottom_layout.addWidget(QLabel("Top results:"))
        bottom_layout.addWidget(self.top_k_combo)

        self.search_btn.clicked.connect(self.handle_search)
        bottom_layout.addWidget(self.search_btn)

        card_layout.addLayout(bottom_layout)

        layout.addWidget(card, 1)

        # Results

        results_title = QLabel("Search Results")
        results_title.setStyleSheet("""
            font-size: 17px;
            font-weight: bold;
        """)

        results_description = QLabel(
            "Your retrieved matches will appear here."
        )
        results_description.setStyleSheet(
            "color: #9295A5; font-size: 11px;"
        )

        layout.addWidget(results_title)
        layout.addWidget(results_description)

        self.results_area = QFrame()
        self.results_area.setMinimumHeight(90)
        self.results_area.setMaximumHeight(280)

        self.results_area.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px dashed #DFE1EB;
                border-radius: 15px;
            }
        """)

        self.results_layout = QVBoxLayout(self.results_area)
        self.results_layout.setContentsMargins(14, 12, 14, 12)
        self.results_layout.setSpacing(8)

        results_icon = QLabel("▧")
        results_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_icon.setStyleSheet("""
            color: #7066E8;
            font-size: 25px;
        """)

        results_text = QLabel("Your results will show up here")
        results_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_text.setStyleSheet(
            "font-weight: bold; font-size: 12px;"
        )

        results_hint = QLabel(
            "Connect the backend to display matching images and captions."
        )
        results_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_hint.setWordWrap(True)
        results_hint.setStyleSheet(
            "color: #9295A5; font-size: 10px;"
        )

        self.results_layout.addWidget(results_icon)
        self.results_layout.addWidget(results_text)
        self.results_layout.addWidget(results_hint)

        layout.addWidget(self.results_area)

        return page

    # -----------------------------------------------------
    # Text Input - Enlarged
    # -----------------------------------------------------

    def create_text_input(self):

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        label = QLabel(
            "Describe the image you want to find"
        )
        label.setStyleSheet("""
            font-weight: bold;
            font-size: 15px;
            color: #252638;
        """)

        layout.addWidget(label)

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText(
            "Describe the image in detail...\n\n"
            "Example: A golden retriever playing with a ball "
            "in a green park on a sunny afternoon."
        )

        self.query_input.setMinimumHeight(190)
        self.query_input.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.query_input.setStyleSheet("""
            QTextEdit {
                background-color: #FBFBFD;
                border: 1px solid #DCDDEA;
                border-radius: 13px;
                padding: 17px;
                font-size: 15px;
                color: #252638;
            }

            QTextEdit:focus {
                border: 2px solid #8279EE;
                background-color: #FFFFFF;
            }
        """)

        layout.addWidget(self.query_input, 1)

        helper = QLabel(
            "Tip: Include objects, colors, actions, and scene details "
            "to describe what you are looking for."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("""
            color: #9295A5;
            font-size: 11px;
            padding-left: 3px;
        """)

        layout.addWidget(helper)

        return widget

    # -----------------------------------------------------
    # Image Input - Enlarged
    # -----------------------------------------------------

    def create_image_input(self):

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        label = QLabel(
            "Upload an image to find matching descriptions"
        )
        label.setStyleSheet("""
            font-weight: bold;
            font-size: 15px;
            color: #252638;
        """)

        layout.addWidget(label)

        self.upload_btn = QPushButton(
            "＋\n\n"
            "Click here to upload an image\n"
            "PNG, JPG, JPEG, WEBP or BMP · Maximum 10 MB"
        )

        self.upload_btn.setMinimumHeight(190)
        self.upload_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #F9F8FF;
                color: #6258E8;
                border: 2px dashed #BDB8F0;
                border-radius: 15px;
                padding: 22px;
                font-size: 15px;
                font-weight: 600;
                line-height: 1.5;
            }

            QPushButton:hover {
                background-color: #F0EFFF;
                border: 2px dashed #8279EE;
            }

            QPushButton:pressed {
                background-color: #EAE8FF;
            }
        """)

        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn, 1)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(210)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #FBFBFD;
                border: 1px solid #E5E7EF;
                border-radius: 12px;
                padding: 8px;
            }
        """)

        self.preview_label.hide()
        layout.addWidget(self.preview_label, 1)

        self.remove_image_btn = QPushButton("✕  Remove image")
        self.remove_image_btn.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.remove_image_btn.setMinimumHeight(40)

        self.remove_image_btn.setStyleSheet("""
            QPushButton {
                color: #DC4545;
                background-color: #FFF4F4;
                border: 1px solid #F8DADA;
                border-radius: 9px;
            }

            QPushButton:hover {
                background-color: #FFE8E8;
            }
        """)

        self.remove_image_btn.clicked.connect(self.clear_image)
        self.remove_image_btn.hide()

        layout.addWidget(self.remove_image_btn)

        return widget

    # -----------------------------------------------------
    # Mode Switching
    # -----------------------------------------------------

    def switch_mode(self, mode):

        self.current_mode = mode
        self.message_label.hide()

        if mode == "text":
            self.input_stack.setCurrentIndex(0)
        else:
            self.input_stack.setCurrentIndex(1)

        self.text_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #6258E8;
                color: white;
            }
        """ if mode == "text" else """
            QPushButton {
                background-color: #F5F5FA;
                color: #777B8C;
            }
        """)

        self.image_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #6258E8;
                color: white;
            }
        """ if mode == "image" else """
            QPushButton {
                background-color: #F5F5FA;
                color: #777B8C;
            }
        """)

        self.set_nav_style(self.text_btn, mode == "text")
        self.set_nav_style(self.image_btn, mode == "image")
        self.set_nav_style(self.overview_btn, False)

    # -----------------------------------------------------
    # Image Upload
    # -----------------------------------------------------

    def upload_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )

        if not file_path:
            return

        try:
            if Path(file_path).stat().st_size > 10 * 1024 * 1024:
                self.show_message(
                    "Image size must be less than 10 MB."
                )
                return
        except OSError:
            self.show_message("Could not access the selected image.")
            return

        pixmap = QPixmap(file_path)

        if pixmap.isNull():
            self.show_message("Could not load the selected image.")
            return

        self.selected_image_path = file_path

        self.update_image_preview(pixmap)

        self.upload_btn.hide()
        self.preview_label.show()
        self.remove_image_btn.show()
        self.message_label.hide()

    def update_image_preview(self, pixmap):

        available_size = self.preview_label.size()

        target_size = QSize(
            max(available_size.width() - 30, 500),
            max(available_size.height() - 20, 180),
        )

        preview = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.preview_label.setPixmap(preview)

    def clear_image(self):

        self.selected_image_path = None

        self.preview_label.clear()
        self.preview_label.hide()

        self.remove_image_btn.hide()
        self.upload_btn.show()

        self.message_label.hide()

    # -----------------------------------------------------
    # Search Validation
    # -----------------------------------------------------

    def handle_search(self):

        if self.current_mode == "text":

            query = self.query_input.toPlainText().strip()

            if not query:
                self.show_message(
                    "Please enter a description before searching."
                )
                return

        else:

            if not self.selected_image_path:
                self.show_message(
                    "Please upload an image before searching."
                )
                return

        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")
        self.message_label.hide()
        self.clear_results("Searching the retrieval gallery...")
        self.search_worker = RetrievalWorker(
            api_base=self.api_base,
            mode=self.current_mode,
            query=query if self.current_mode == "text" else "",
            image_path=self.selected_image_path,
            top_k=int(self.top_k_combo.currentText()),
            model=self.model_type,
        )
        self.search_worker.succeeded.connect(self.on_search_success)
        self.search_worker.failed.connect(self.on_search_failure)
        self.search_worker.start()

    def clear_results(self, message):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        label = QLabel(message)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #777B8C; padding: 12px; font-size: 12px;")
        self.results_layout.addWidget(label)

    def on_search_success(self, payload):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("⌕  Search  →")
        results = payload.get("results") or []
        self.clear_results(
            f"{len(results)} result(s) · {payload.get('elapsed_ms', 0):.1f} ms · {payload.get('model', self.model_type)}"
        )
        for item in results[:5]:
            kind = item.get("type", "result")
            value = item.get("caption") or item.get("image_id") or item.get("result") or "Matching item"
            if isinstance(value, dict):
                value = value.get("caption") or value.get("image_id") or json.dumps(value, ensure_ascii=False)
            score = item.get("similarity", item.get("score", 0.0))
            label = QLabel(f"#{item.get('rank', '?')}  [{kind}]  {str(value)}\nSimilarity: {float(score):.4f}")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setStyleSheet("background: #F8F8FC; border: 1px solid #E8EAF1; border-radius: 8px; padding: 9px; font-size: 11px;")
            self.results_layout.addWidget(label)
        if not results:
            self.clear_results("No matching results were returned. Try a different query or image.")

    def on_search_failure(self, error):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("⌕  Search  →")
        self.show_message(
            f"Could not complete retrieval. Check that the FastAPI backend is running at {self.api_base}. Details: {error}"
        )
        self.clear_results("No results available.")

    def show_message(self, message):

        self.message_label.setText(message)
        self.message_label.show()

    # -----------------------------------------------------
    # General Navigation
    # -----------------------------------------------------

    def show_history(self):

        QMessageBox.information(
            self,
            "Search History",
            "Search history will be available after backend integration."
        )

    def show_about(self):

        QMessageBox.information(
            self,
            "About VisionMatch AI",
            "VisionMatch AI is a cross-modal image-text retrieval "
            "system powered by deep learning.\n\n"
            "It supports Text-to-Image and Image-to-Text retrieval."
        )
