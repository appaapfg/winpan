import os

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

try:
    from panopticon.config import SETTINGS_FILE
except ImportError:  # fallback when running as script
    from config import SETTINGS_FILE


class SegmentationOpacitySelectorWidget(QDialog):
    def __init__(self, tr_func, parent=None):
        super().__init__(parent)
        self.tr = tr_func
        self.setWindowTitle(self.tr("segmentation_opacity_dialog_title"))
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)

        self.slider_label = QLabel(self.tr("segmentation_opacity_label"))
        main_layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)  # 0.01
        self.slider.setMaximum(100)  # 1.00
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self._update_label)
        main_layout.addWidget(self.slider)

        self.value_label = QLabel("0.00")
        main_layout.addWidget(self.value_label)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton(self.tr("save_button"))
        self.cancel_button = QPushButton(self.tr("cancel_button"))
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Load persisted setting (defaults applied if unavailable)
        self.load_settings()
        self._update_label(self.slider.value())

    def showEvent(self, event):
        # Remember previous value to restore if dialog is cancelled
        self._previous_opacity_on_show = self.get_opacity()
        super().showEvent(event)

    def _update_label(self, value):
        self.value_label.setText(f"{value / 100:.2f}")

    def get_opacity(self):
        return self.slider.value() / 100.0

    # --- Settings persistence -------------------------------------------------
    def load_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    settings = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                print(self.tr("status_error_decoding_yaml_default_opacity").format(file=SETTINGS_FILE))

        # Default to 0.9 (90 in slider units) if not found or invalid
        opacity = settings.get("segmentation_opacity", 0.9)
        # Clamp just in case
        try:
            opacity = float(opacity)
        except (TypeError, ValueError):
            opacity = 0.9
        opacity = max(0.01, min(opacity, 1.0))
        self.slider.setValue(int(opacity * 100))

    def save_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    existing_settings = yaml.safe_load(f)
                if existing_settings is not None:
                    settings = existing_settings
            except yaml.YAMLError:
                print(self.tr("status_error_decoding_yaml_overwriting").format(file=SETTINGS_FILE))

        settings["segmentation_opacity"] = self.get_opacity()

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(settings, f, indent=4)
        except OSError as e:
            print(self.tr("status_error_saving_settings_to_file").format(file=SETTINGS_FILE, e=e))

    def accept(self):
        self.save_settings()
        super().accept()

    def reject(self):
        # Restore previous value (no save)
        if hasattr(self, "_previous_opacity_on_show"):
            self.slider.setValue(int(self._previous_opacity_on_show * 100))
        super().reject()

    # -------------------------------------------------------------------------
    def retranslate_ui(self):
        self.setWindowTitle(self.tr("segmentation_opacity_dialog_title"))
        self.slider_label.setText(self.tr("segmentation_opacity_label"))
        self.save_button.setText(self.tr("save_button"))
        self.cancel_button.setText(self.tr("cancel_button"))
