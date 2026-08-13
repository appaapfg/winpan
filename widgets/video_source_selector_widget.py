from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

try:
    from panopticon.config import SETTINGS_FILE
except ImportError:
    from config import SETTINGS_FILE


DEFAULT_VIDEO_SOURCES = ""


class VideoSourceSelectorWidget(QDialog):
    def __init__(self, tr_func, parent=None, config_key_name="video_sources"):
        super().__init__(parent)
        self.tr = tr_func
        self.config_key_name = config_key_name
        self.setWindowTitle(self.tr("configure_video_sources_action"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # Add a detailed description label at the top
        description_label = QLabel(self.tr("video_sources_description"))
        description_label.setWordWrap(True)
        self.layout.addWidget(description_label)

        # Video sources input
        self.sources_layout = QHBoxLayout()
        self.sources_label = QLabel(self.tr("video_source_directory_label"))
        self.sources_input = QLineEdit()
        self.sources_input.setPlaceholderText("Select a directory containing video files")
        self.browse_button = QPushButton(self.tr("browse_button"))
        self.sources_layout.addWidget(self.sources_label)
        self.sources_layout.addWidget(self.sources_input)
        self.sources_layout.addWidget(self.browse_button)
        self.layout.addLayout(self.sources_layout)

        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.save_button = QPushButton(self.tr("save_button"))
        self.cancel_button = QPushButton(self.tr("cancel_button"))
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.buttons_layout)

        self.browse_button.clicked.connect(self._select_directory)
        self.save_button.clicked.connect(self._save_settings)
        self.cancel_button.clicked.connect(self.reject)

        self.load_settings()

    def _select_directory(self):
        from PySide6.QtWidgets import QFileDialog

        dir_path = QFileDialog.getExistingDirectory(self, self.tr("select_video_source_directory_title"), "")
        if dir_path:
            self.sources_input.setText(dir_path)

    def load_settings(self):
        settings = self._load_yaml_settings()
        sources = settings.get(self.config_key_name, DEFAULT_VIDEO_SOURCES)
        self.sources_input.setText(str(sources))

    def _save_settings(self):
        try:
            sources = self.sources_input.text() if self.sources_input.text() else DEFAULT_VIDEO_SOURCES

            settings = self._load_yaml_settings()
            settings[self.config_key_name] = sources
            self._write_yaml_settings(settings)
            self.accept()
        except Exception:
            QMessageBox.critical(self, self.tr("error_title"), self.tr("error_saving_settings"))

    def _load_yaml_settings(self):
        settings = {}
        if Path(SETTINGS_FILE).exists():
            with open(SETTINGS_FILE) as f:
                settings = yaml.safe_load(f) or {}
        return settings

    def _write_yaml_settings(self, settings):
        with open(SETTINGS_FILE, "w") as f:
            yaml.safe_dump(settings, f, default_flow_style=False)

    def get_video_sources(self):
        return self.sources_input.text() if self.sources_input.text() else DEFAULT_VIDEO_SOURCES
