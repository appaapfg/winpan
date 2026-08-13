import os

import yaml
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from panopticon.config import SETTINGS_FILE
from panopticon.translations_data import TRANSLATIONS


class LanguageActionGroup(QMenu):
    def __init__(
        self,
        main_window,
        tr_function,
        set_language_callback,
        status_bar,
    ):
        super().__init__()
        self.main_window = main_window
        self.tr = tr_function
        self.set_language_callback = set_language_callback
        self.status_bar = status_bar

        self.action_group = QActionGroup(self.main_window)

        self.english_action = QAction()
        self.english_action.setCheckable(True)
        self.english_action.triggered.connect(lambda: self._on_language_selected("en"))
        self.addAction(self.english_action)
        self.action_group.addAction(self.english_action)

        self.greek_action = QAction()
        self.greek_action.setCheckable(True)
        self.greek_action.triggered.connect(lambda: self._on_language_selected("el"))
        self.addAction(self.greek_action)
        self.action_group.addAction(self.greek_action)

        self.load_language_setting()
        self.set_language(self.main_window.current_language)
        self.retranslate_ui()

    def save_language_setting(self, language_code):
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
                print(f"Error reading settings file for saving language: {e}")
                if self.status_bar:
                    self.status_bar.showMessage(self.tr("status_error_reading_settings", e=e), 5000)
                all_settings = {}

        all_settings["language"] = language_code
        self.main_window.current_language = language_code

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
            if self.status_bar:
                lang_name = "English" if language_code == "en" else "Ελληνικά"
                self.status_bar.showMessage(self.tr("status_language_changed", lang=lang_name), 2000)
        except Exception as e:
            print(f"Error writing language setting: {e}")
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_error_saving_lang", err=e), 5000)

    def load_language_setting(self):
        default_lang = "en"
        loaded_lang = default_lang

        if not os.path.exists(SETTINGS_FILE):
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_lang_not_found"), 3000)
            self.main_window.current_language = default_lang
            return

        try:
            with open(SETTINGS_FILE) as f:
                all_settings = yaml.safe_load(f)
                if all_settings and isinstance(all_settings, dict) and "language" in all_settings:
                    lang_code = all_settings.get("language")
                    if lang_code in TRANSLATIONS:
                        loaded_lang = lang_code
                        if self.status_bar:
                            lang_name = "English" if loaded_lang == "en" else "Ελληνικά"
                            self.status_bar.showMessage(self.tr("status_lang_loaded", lang=lang_name), 2000)
                    else:
                        if self.status_bar:
                            self.status_bar.showMessage(self.tr("status_lang_not_found"), 3000)
                else:
                    if self.status_bar:
                        self.status_bar.showMessage(self.tr("status_lang_not_found"), 3000)
        except Exception as e:
            print(f"Error loading language setting: {e}")
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_error_loading_lang", err=e), 5000)

        self.main_window.current_language = loaded_lang

    def _on_language_selected(self, lang_code: str):
        self.set_language_callback(lang_code)

    def retranslate_ui(self):
        self.setTitle(self.tr("language_menu"))
        self.english_action.setText(self.tr("english_action"))
        self.greek_action.setText(self.tr("greek_action"))

    def get_selected_language(self) -> str:
        if self.english_action.isChecked():
            return "en"
        if self.greek_action.isChecked():
            return "el"
        return "en"  # default

    def set_language(self, lang_code: str):
        if lang_code == "en":
            self.english_action.setChecked(True)
        elif lang_code == "el":
            self.greek_action.setChecked(True)
