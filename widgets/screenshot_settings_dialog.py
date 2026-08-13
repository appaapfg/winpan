import os
import sys

import yaml
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from panopticon.config import SETTINGS_FILE

DEFAULT_SCREENSHOT_PATH = "panopticon_screenshots"


class ScreenshotSettingsDialog(QDialog):
    def __init__(self, tr, parent=None):
        super().__init__(parent)
        self.tr = tr
        self.setWindowTitle(self.tr("screenshot_settings_dialog_title"))
        # self.resize(500, 400)

        self.layout = QVBoxLayout(self)

        # Add a detailed description label at the top
        self.description_label = QLabel(self.tr("screenshot_settings_dialog_description"))
        self.description_label.setWordWrap(True)
        self.layout.addWidget(self.description_label)

        # Path selection
        self.path_layout = QHBoxLayout()
        self.path_label = QLabel(self.tr("screenshot_settings_label"))
        self.path_input = QLineEdit()
        self.browse_button = QPushButton(self.tr("browse_button"))
        self.browse_button.clicked.connect(self._browse_directory)

        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_input)
        self.path_layout.addWidget(self.browse_button)
        self.layout.addLayout(self.path_layout)

        # Cooldown selection
        self.cooldown_layout = QHBoxLayout()
        self.cooldown_label = QLabel(self.tr("auto_screenshot_cooldown_label"))
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(500, 10000)
        self.cooldown_spin.setValue(2000)
        self.cooldown_spin.setSuffix(" ms")
        self.cooldown_spin.setSingleStep(100)

        self.cooldown_layout.addWidget(self.cooldown_label)
        self.cooldown_layout.addWidget(self.cooldown_spin)
        self.layout.addLayout(self.cooldown_layout)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.ok_button = QPushButton(self.tr("ok_button"))
        self.ok_button.clicked.connect(self.accept)  # Connect to accept slot
        self.cancel_button = QPushButton(self.tr("cancel_button"))
        self.cancel_button.clicked.connect(self.reject)  # Connect to reject slot

        self.button_layout.addStretch()
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        self.load_settings()

    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("select_directory_title"), self.path_input.text())
        if directory:
            self.path_input.setText(directory)

    def get_selected_path(self):
        return self.path_input.text()

    def get_cooldown(self):
        return self.cooldown_spin.value()

    def load_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        settings = content
            except Exception as e:
                # Consider logging this error or showing a status message
                print(f"Error reading settings file: {e}")

        path = settings.get("screenshot_path", DEFAULT_SCREENSHOT_PATH)
        self.path_input.setText(path)

        cooldown = settings.get("auto_screenshot_cooldown", 2000)
        self.cooldown_spin.setValue(cooldown)

    def save_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        settings = content
            except Exception as e:
                print(f"Error reading settings file for saving: {e}")
                # Decide if we want to overwrite or stop, for now, it will create/overwrite

        settings["screenshot_path"] = self.get_selected_path()
        settings["auto_screenshot_cooldown"] = self.get_cooldown()

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            # Consider showing an error to the user
            print(f"Error writing screenshot path setting: {e}")

    # Override accept to save settings
    def accept(self):
        self.save_settings()
        super().accept()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("screenshot_settings_dialog_title"))
        self.description_label.setText(self.tr("screenshot_settings_dialog_description"))
        self.path_label.setText(self.tr("screenshot_settings_label"))
        self.cooldown_label.setText(self.tr("auto_screenshot_cooldown_label"))
        self.browse_button.setText(self.tr("browse_button"))
        self.ok_button.setText(self.tr("ok_button"))
        self.cancel_button.setText(self.tr("cancel_button"))
        # The title for QFileDialog.getExistingDirectory is set when _browse_directory is called.
        # If the dialog needs to be re-opened for retranslation of that title,
        # it would require more complex handling, or creating it on-the-fly each time


def get_selected_screenshot_path():
    """
    Returns the currently selected screenshot path from the settings file.
    If the settings file or the key does not exist, returns DEFAULT_SCREENSHOT_PATH.
    """
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    return content.get("screenshot_path", DEFAULT_SCREENSHOT_PATH)
        except Exception as e:
            print(f"Error reading settings file: {e}")
    return DEFAULT_SCREENSHOT_PATH


def get_auto_screenshot_cooldown():
    """
    Returns the currently selected auto screenshot cooldown from the settings file.
    If the settings file or the key does not exist, returns 2000 ms.
    """
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    return content.get("auto_screenshot_cooldown", 2000)
        except Exception as e:
            print(f"Error reading settings file: {e}")
    return 2000


# Example usage (for testing)
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dummy tr function for testing
    def dummy_tr(key, **kwargs):
        translations = {
            "screenshot_settings_dialog_title": "Screenshot Settings Configuration",
            "screenshot_settings_label": "Path:",
            "auto_screenshot_cooldown_label": "Auto Screenshot Cooldown:",
            "browse_button": "Browse...",
            "ok_button": "OK",
            "cancel_button": "Cancel",
            "select_directory_title": "Select Screenshot Directory",
        }
        return translations.get(key, key).format(**kwargs) if kwargs else translations.get(key, key)

    # Create a dummy SETTINGS_FILE for testing if it doesn't exist
    # and ensure DEFAULT_SCREENSHOT_PATH is usable
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "w") as f:
            yaml.dump({"screenshot_path": DEFAULT_SCREENSHOT_PATH, "auto_screenshot_cooldown": 2000}, f)  # Use imported DEFAULT_SCREENSHOT_PATH
    elif not isinstance(yaml.safe_load(open(SETTINGS_FILE)), dict) or yaml.safe_load(open(SETTINGS_FILE)).get("screenshot_path") is None:
        with open(SETTINGS_FILE) as f_read:
            current_settings = yaml.safe_load(f_read)
            if not isinstance(current_settings, dict):
                current_settings = {}
        current_settings["screenshot_path"] = DEFAULT_SCREENSHOT_PATH
        current_settings["auto_screenshot_cooldown"] = 2000
        with open(SETTINGS_FILE, "w") as f_write:
            yaml.dump(current_settings, f_write)

    dialog = ScreenshotSettingsDialog(tr=dummy_tr)
    if dialog.exec():
        print(f"Selected path: {dialog.get_selected_path()}")
        print(f"Selected cooldown: {dialog.get_cooldown()} ms")
        print(f"Settings saved to {SETTINGS_FILE}")
    else:
        print("Dialog cancelled.")

    sys.exit(app.exec())
