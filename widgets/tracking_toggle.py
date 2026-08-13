"""Tracking toggle widget for enabling/disabling object tracking."""

import os

import yaml
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu

from panopticon.config import SETTINGS_FILE

SETTINGS_KEY = "enable_tracking"
DEFAULT_ENABLED = False


class TrackingToggle(QMenu):
    """A menu containing a toggle action for enabling/disabling object tracking.

    This class manages the tracking enabled state, including saving and loading
    the setting from the settings file.
    """

    def __init__(self, parent, tr_func, status_bar):
        """Initialize the TrackingToggle.

        Args:
            parent: The parent widget.
            tr_func: The translation function.
            status_bar: The status bar for displaying messages.
        """
        super().__init__(parent)
        self.tr = tr_func
        self.status_bar = status_bar
        self._callbacks = []

        self.toggle_action = QAction("", parent)
        self.toggle_action.setCheckable(True)
        self.toggle_action.setChecked(self._load_setting())
        self.toggle_action.setShortcut(QKeySequence("Ctrl+T"))
        self.toggle_action.triggered.connect(self._on_toggled)

    def get_action(self) -> QAction:
        """Return the toggle action to be added to a menu.

        Returns:
            QAction: The tracking toggle action.
        """
        return self.toggle_action

    def is_enabled(self) -> bool:
        """Check if tracking is currently enabled.

        Returns:
            bool: True if tracking is enabled, False otherwise.
        """
        return self.toggle_action.isChecked()

    def add_callback(self, callback):
        """Add a callback to be called when tracking is toggled.

        Args:
            callback: A function that takes a boolean (enabled state) as argument.
        """
        self._callbacks.append(callback)

    def _load_setting(self) -> bool:
        """Load the tracking enabled setting from settings file.

        Returns:
            bool: The saved tracking state, or DEFAULT_ENABLED if not found.
        """
        if not os.path.exists(SETTINGS_FILE):
            return DEFAULT_ENABLED
        try:
            with open(SETTINGS_FILE) as f:
                settings = yaml.safe_load(f)
                if settings and isinstance(settings, dict):
                    return settings.get(SETTINGS_KEY, DEFAULT_ENABLED)
        except Exception as e:
            print(f"Error loading tracking setting: {e}")
        return DEFAULT_ENABLED

    def _save_setting(self, enabled: bool):
        """Save the tracking enabled setting to settings file.

        Args:
            enabled: The tracking enabled state to save.
        """
        all_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    loaded_content = yaml.safe_load(f)
                    if isinstance(loaded_content, dict):
                        all_settings = loaded_content
            except Exception as e:
                print(f"Error reading settings file: {e}")
        all_settings[SETTINGS_KEY] = enabled
        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving tracking setting: {e}")

    def _on_toggled(self):
        """Handle tracking toggle action."""
        enabled = self.toggle_action.isChecked()
        self._save_setting(enabled)

        if enabled:
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_tracking_enabled"), 2000)
        else:
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_tracking_disabled"), 2000)

        for callback in self._callbacks:
            try:
                callback(enabled)
            except Exception as e:
                print(f"Error in tracking toggle callback: {e}")

    def retranslate_ui(self):
        """Update the UI text for the current language."""
        self.toggle_action.setText(self.tr("enable_tracking_action"))
