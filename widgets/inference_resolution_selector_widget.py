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


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 640


class InferenceResolutionSelectorWidget(QDialog):
    def __init__(self, tr_func, parent=None, config_key_name="detection_inference_resolution"):
        super().__init__(parent)
        self.tr = tr_func
        self.config_key_name = config_key_name
        self.setWindowTitle(self.tr("configure_inference_resolution_action"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # Add a detailed description label at the top
        description_label = QLabel(self.tr("inference_resolution_description"))
        description_label.setWordWrap(True)
        self.layout.addWidget(description_label)

        # Width input
        self.width_layout = QHBoxLayout()
        self.width_label = QLabel(self.tr("inference_resolution_width_label"))
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText(str(DEFAULT_WIDTH))
        self.width_input.setValidator(self.create_int_validator())
        self.width_layout.addWidget(self.width_label)
        self.width_layout.addWidget(self.width_input)
        self.layout.addLayout(self.width_layout)

        # Height input
        self.height_layout = QHBoxLayout()
        self.height_label = QLabel(self.tr("inference_resolution_height_label"))
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText(str(DEFAULT_HEIGHT))
        self.height_input.setValidator(self.create_int_validator())
        self.height_layout.addWidget(self.height_label)
        self.height_layout.addWidget(self.height_input)
        self.layout.addLayout(self.height_layout)

        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.save_button = QPushButton(self.tr("save_button"))
        self.cancel_button = QPushButton(self.tr("cancel_button"))
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.buttons_layout)

        self.save_button.clicked.connect(self._save_settings)
        self.cancel_button.clicked.connect(self.reject)

        self.load_settings()

    def create_int_validator(self):
        from PySide6.QtGui import QIntValidator

        return QIntValidator(1, 9999, self)

    def load_settings(self):
        settings = self._load_yaml_settings()
        width = settings.get(self.config_key_name, {}).get("width", DEFAULT_WIDTH)
        height = settings.get(self.config_key_name, {}).get("height", DEFAULT_HEIGHT)
        self.width_input.setText(str(width))
        self.height_input.setText(str(height))

    def _save_settings(self):
        try:
            width = int(self.width_input.text()) if self.width_input.text() else DEFAULT_WIDTH
            height = int(self.height_input.text()) if self.height_input.text() else DEFAULT_HEIGHT

            settings = self._load_yaml_settings()
            if self.config_key_name not in settings:
                settings[self.config_key_name] = {}
            settings[self.config_key_name]["width"] = width
            settings[self.config_key_name]["height"] = height
            self._write_yaml_settings(settings)
            self.accept()
        except ValueError:
            QMessageBox.warning(self, self.tr("invalid_input_title"), self.tr("invalid_resolution_input_message"))
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

    def get_resolution(self):
        width = int(self.width_input.text()) if self.width_input.text() else DEFAULT_WIDTH
        height = int(self.height_input.text()) if self.height_input.text() else DEFAULT_HEIGHT
        return height, width
