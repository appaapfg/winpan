import os

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,  # QMenu removed as it's not directly used by ThemeActionGroup
)

from panopticon.config import SETTINGS_FILE

DEFAULT_THEME = "light"


class ThemeActionGroup(QActionGroup):
    """
    Manages theme selection actions, applies themes, and handles saving/loading
    theme settings.
    """

    def __init__(
        self,
        parent_menu_bar_or_menu,  # QMenuBar or QMenu to add the theme menu to
        main_window,  # Parent window, typically QMainWindow
        app_instance,  # QApplication instance
        tr_func,  # Translation function (e.g., QCoreApplication.translate)
        status_bar_instance=None,  # Optional QStatusBar instance for messages
    ):
        super().__init__(main_window)
        self.main_window = main_window
        self.app_instance = app_instance
        self.tr_func = tr_func
        self.status_bar_instance = status_bar_instance

        self.theme_menu = parent_menu_bar_or_menu.addMenu("")

        self.setExclusive(True)

        self.light_theme_action = QAction("", self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.setData("light")
        self.theme_menu.addAction(self.light_theme_action)
        self.addAction(self.light_theme_action)

        self.dark_theme_action = QAction("", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setData("dark")
        self.theme_menu.addAction(self.dark_theme_action)
        self.addAction(self.dark_theme_action)

        self.deep_dark_theme_action = QAction("", self)
        self.deep_dark_theme_action.setCheckable(True)
        self.deep_dark_theme_action.setData("deep_dark")
        self.theme_menu.addAction(self.deep_dark_theme_action)
        self.addAction(self.deep_dark_theme_action)

        self.triggered.connect(self._on_theme_selected)

        self._load_initial_theme_and_apply()

    def _on_theme_selected(self, action):
        theme_key = action.data()
        self._save_theme_setting_internal(theme_key)
        self._apply_theme_internal(theme_key)

        theme_display_name_key = f"{theme_key}_theme_action"
        if self.status_bar_instance:
            try:
                translated_theme_name = self.tr_func(theme_display_name_key)
                status_message = self.tr_func("status_theme_selected", theme_name=translated_theme_name)
            except Exception:  # Fallback if tr_func or key is problematic for display name
                status_message = f"Theme set to {theme_key}"

            self.status_bar_instance.showMessage(status_message, 2000)

    def _apply_theme_internal(self, theme_key):
        if theme_key == "dark":
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.WindowText,
                Qt.GlobalColor.white,
            )
            dark_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.AlternateBase,
                QColor(53, 53, 53),
            )
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.ToolTipBase,
                Qt.GlobalColor.white,
            )
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.ToolTipText,
                Qt.GlobalColor.white,
            )
            dark_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Text, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.ButtonText,
                Qt.GlobalColor.white,
            )
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.BrightText,
                Qt.GlobalColor.red,
            )
            dark_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Link, QColor(42, 130, 218))
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.Highlight,
                QColor(42, 130, 218),
            )
            dark_palette.setColor(
                QPalette.ColorGroup.All,
                QPalette.ColorRole.HighlightedText,
                Qt.GlobalColor.black,
            )
            self.app_instance.setPalette(dark_palette)
        elif theme_key == "deep_dark":
            oled_palette = QPalette()
            # True black for OLED, with high contrast text
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Window, QColor(0, 0, 0))
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Base, QColor(10, 10, 10))
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.AlternateBase, QColor(0, 0, 0))
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Text, Qt.GlobalColor.white)
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Button, QColor(0, 0, 0))
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Link, QColor(42, 130, 218))
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            oled_palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            self.app_instance.setPalette(oled_palette)
        else:  # "light" or default
            self.app_instance.setPalette(QApplication.style().standardPalette())

    def _save_theme_setting_internal(self, theme_key):
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
                print(f"Error reading settings file for saving theme: {e}")
                if self.status_bar_instance:
                    self.status_bar_instance.showMessage(self.tr_func("status_error_reading_settings", e=str(e)), 5000)
                all_settings = {}

        all_settings["theme"] = theme_key

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error writing theme setting: {e}")
            if self.status_bar_instance:
                self.status_bar_instance.showMessage(self.tr_func("status_error_saving_theme", err=str(e)), 5000)

    def _load_initial_theme_and_apply(self):
        loaded_theme_key = DEFAULT_THEME

        if not os.path.exists(SETTINGS_FILE):
            if self.status_bar_instance:
                self.status_bar_instance.showMessage(
                    self.tr_func("status_settings_file_not_found_loading_default_theme"),
                    2000,
                )
        else:
            try:
                with open(SETTINGS_FILE) as f:
                    all_settings = yaml.safe_load(f)
                    if all_settings and isinstance(all_settings, dict) and "theme" in all_settings:
                        theme_from_file = all_settings.get("theme")
                        if theme_from_file in ["light", "dark"]:
                            loaded_theme_key = theme_from_file
                            if self.status_bar_instance:
                                display_name_key = f"{loaded_theme_key}_theme_action"
                                try:
                                    translated_theme_name = self.tr_func(display_name_key)
                                    status_message = self.tr_func(
                                        "status_theme_loaded",
                                        theme_name=translated_theme_name,
                                    )
                                except Exception:  # Fallback
                                    status_message = f"Loaded theme: {loaded_theme_key}"
                                self.status_bar_instance.showMessage(status_message, 2000)
                        else:
                            if self.status_bar_instance:
                                self.status_bar_instance.showMessage(
                                    self.tr_func("status_invalid_theme_in_settings_loading_default"),
                                    2000,
                                )
                    else:
                        if self.status_bar_instance:
                            self.status_bar_instance.showMessage(
                                self.tr_func("status_theme_not_found_in_settings_loading_default"),
                                2000,
                            )
            except Exception as e:
                print(f"Error loading theme setting: {e}")
                if self.status_bar_instance:
                    self.status_bar_instance.showMessage(
                        self.tr_func("status_error_loading_theme_settings", err=str(e)),
                        5000,
                    )

        self._apply_theme_internal(loaded_theme_key)

        if loaded_theme_key == "light":
            self.light_theme_action.setChecked(True)
        elif loaded_theme_key == "dark":
            self.dark_theme_action.setChecked(True)
        elif loaded_theme_key == "deep_dark":
            self.deep_dark_theme_action.setChecked(True)

    def retranslate_ui(self):
        """Updates the translatable texts for the theme menu and actions."""
        self.theme_menu.setTitle(self.tr_func("color_theme_menu"))
        self.light_theme_action.setText(self.tr_func("light_theme_action"))
        self.dark_theme_action.setText(self.tr_func("dark_theme_action"))
        self.deep_dark_theme_action.setText(self.tr_func("deep_dark_theme_action"))
