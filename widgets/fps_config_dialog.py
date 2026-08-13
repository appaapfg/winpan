import os

import yaml
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from panopticon.config import SETTINGS_FILE


class FPSConfigDialog(QDialog):
    def __init__(self, parent, tr_function):
        super().__init__(parent)
        self.tr = tr_function

        self.setWindowTitle(self.tr("fps_config_dialog_title"))
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        info_label = QLabel(self.tr("fps_config_dialog_info"))
        info_label.setWordWrap(True)
        form_layout.addRow(info_label)

        self.fps_input = QLineEdit()
        self.fps_input.setPlaceholderText(self.tr("fps_config_dialog_fps_placeholder"))
        # Consider adding QDoubleValidator if strict float input is needed
        # from PySide6.QtGui import QDoubleValidator
        # self.fps_input.setValidator(QDoubleValidator(0.1, 120.0, 2, self))

        form_layout.addRow(self.tr("fps_config_dialog_fps_label"), self.fps_input)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Save).setText(self.tr("save_button"))
        button_box.button(QDialogButtonBox.Cancel).setText(self.tr("cancel_button"))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as f:
                    settings = yaml.safe_load(f)
                    if settings and "fps_setting" in settings:
                        self.fps_input.setText(str(settings.get("fps_setting", "")))
        except Exception as e:
            print(f"Error loading FPS settings: {e}")
            # Optionally, inform the user via a status message if a status_bar_instance is available

    def save_settings(self):
        all_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    loaded_content = yaml.safe_load(f)
                    if isinstance(loaded_content, dict):
                        all_settings = loaded_content
                    else:
                        all_settings = {}
            except Exception as e:
                print(f"Error reading settings file for saving FPS: {e}")
                all_settings = {}

        fps_text = self.fps_input.text()
        try:
            fps_value = float(fps_text)
            all_settings["fps_setting"] = fps_value
        except ValueError:
            print(f"Invalid FPS value: {fps_text}. Not saving FPS.")
            # Optionally, show an error to the user
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("status_fps_value_error"))
            return False

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error saving FPS settings: {e}")
            return False

    def accept(self):
        if self.save_settings():
            super().accept()
        # else: an error message could be shown here

    def get_fps_value(self):
        try:
            return float(self.fps_input.text())
        except ValueError:
            return None


def get_configured_fps(tr_function=None, status_bar_instance=None):
    """Safely retrieves the configured FPS setting from the YAML file."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                settings = yaml.safe_load(f)
                if settings and "fps_setting" in settings:
                    fps = float(settings["fps_setting"])
                    if fps > 0:
                        return fps
    except (ValueError, TypeError, Exception) as e:
        # Using tr_function if available for logging/status
        message = f"Could not read or parse FPS setting, using default. Error: {e}"
        if tr_function and status_bar_instance:
            status_bar_instance.showMessage(tr_function("status_error_reading_fps_settings"), 5000)
        else:
            print(message)
    return 25.0  # Default FPS


def calculate_fps_interval(tr_function, status_bar_instance=None):
    """Calculates the timer interval in milliseconds based on the configured FPS."""
    fps = get_configured_fps(tr_function, status_bar_instance)
    if fps > 0:
        return int(1000 / fps)
    return 1000  # Default to 1 FPS (1000 ms interval) if not configured or invalid
