import os
import yaml
from PySide6.QtGui import QAction
from panopticon.config import SETTINGS_FILE

class PersistentCheckableAction(QAction):
    """A QAction that persists its checked state to a settings file."""

    def __init__(self, text, parent, settings_key, default_value=False):
        """Initialize the PersistentCheckableAction.

        Args:
            text (str): The text to display for the action.
            parent (QObject): The parent object.
            settings_key (str): The key to use for saving/loading the setting.
            default_value (bool): The default checked state if no setting is found.
        """
        super().__init__(text, parent)
        self.settings_key = settings_key
        self.default_value = default_value
        self.setCheckable(True)
        
        # Load setting and set initial state
        initial_state = self._load_setting()
        self.setChecked(initial_state)
        
        # Connect toggled signal to save setting
        self.toggled.connect(self._on_toggled)

    def _load_setting(self) -> bool:
        """Load the checked state from settings file."""
        if not os.path.exists(SETTINGS_FILE):
            return self.default_value
        try:
            with open(SETTINGS_FILE) as f:
                settings = yaml.safe_load(f)
                if settings and isinstance(settings, dict):
                    return settings.get(self.settings_key, self.default_value)
        except Exception as e:
            print(f"Error loading setting {self.settings_key}: {e}")
        return self.default_value

    def _save_setting(self, enabled: bool):
        """Save the checked state to settings file."""
        all_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    loaded_content = yaml.safe_load(f)
                    if isinstance(loaded_content, dict):
                        all_settings = loaded_content
            except Exception as e:
                print(f"Error reading settings file: {e}")
        
        all_settings[self.settings_key] = enabled
        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving setting {self.settings_key}: {e}")

    def _on_toggled(self, checked: bool):
        """Handle toggle event."""
        self._save_setting(checked)
