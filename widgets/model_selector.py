import os

import yaml
from PySide6.QtGui import QAction, QActionGroup

from panopticon.config import SETTINGS_FILE
from panopticon.my_inference.detect import Detector
from panopticon.my_inference.segment import Segmenter


class ModelSelector(QActionGroup):
    """Generic model selector for detection or segmentation models.

    Args:
        parent: Qt parent.
        tr_func: translation function.
        status_bar: optional status bar for messages.
        auto_load_save: whether to auto load/save selection to YAML settings.
        mode: "detection" or "segmentation".
    """

    DETECTION_MODEL_LIST = [None] + Detector.SUPPORTED_MODELS
    SEGMENTATION_MODEL_LIST = [None] + Segmenter.SUPPORTED_MODELS
    DEFAULT_DETECTION_MODEL = None
    DEFAULT_SEGMENTATION_MODEL = None

    VALID_MODES = {"detection", "segmentation"}

    def __init__(self, parent, tr_func, status_bar, auto_load_save=True, mode: str = "detection"):
        super().__init__(parent)
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of {self.VALID_MODES}.")

        self.mode = mode
        self.tr = tr_func
        self.status_bar = status_bar
        self.auto_load_save = auto_load_save
        self.actions: dict = {}  # To store actions by key for easy access

        # Configuration depending on mode
        if self.mode == "detection":
            self._model_list = self.DETECTION_MODEL_LIST
            self._yaml_key = "detection_model"
            self._default_model = self.DEFAULT_DETECTION_MODEL
        else:  # segmentation
            self._model_list = self.SEGMENTATION_MODEL_LIST
            self._yaml_key = "segmentation_model"
            self._default_model = self.DEFAULT_SEGMENTATION_MODEL

        self.setExclusive(True)

        # Create QAction entries for each model choice
        for key in self._model_list:
            action = QAction(str(key), self)  # Use key directly for model name
            action.setCheckable(True)
            action.setData(key)
            self.addAction(action)
            self.actions[key] = action

        self.triggered.connect(self._on_model_selected)
        if self.auto_load_save:
            self.load_setting()

    def _on_model_selected(self, action):
        model_key = action.data()
        if self.auto_load_save:
            self.save_setting(model_key)
        if self.status_bar:
            # Avoid passing unexpected keyword args into translation helper.
            self.status_bar.showMessage(f"{self.tr('status_model_selected')}: {model_key}", 2000)

    def save_setting(self, model_key):
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
                print(f"Error reading settings file for saving detection model: {e}")
                if self.status_bar:
                    self.status_bar.showMessage(self.tr("status_error_reading_settings"), 5000)
                all_settings = {}

        # Persist under mode-specific key
        all_settings[self._yaml_key] = model_key

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error writing model setting: {e}")
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_error_saving_model"), 5000)

    def load_setting(self):
        loaded_model_key = self._default_model

        if not os.path.exists(SETTINGS_FILE):
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_model_not_found"), 2000)
        else:
            try:
                with open(SETTINGS_FILE) as f:
                    all_settings = yaml.safe_load(f)
                    if all_settings and isinstance(all_settings, dict) and self._yaml_key in all_settings:
                        model_from_file = all_settings.get(self._yaml_key)
                        if model_from_file in self.actions:  # Check if model key is valid
                            loaded_model_key = model_from_file
                            if self.status_bar:
                                self.status_bar.showMessage(f"{self.tr('status_model_loaded')}: {loaded_model_key}", 2000)
                        else:
                            if self.status_bar:
                                self.status_bar.showMessage(self.tr("status_model_not_found"), 2000)
                    else:
                        if self.status_bar:
                            self.status_bar.showMessage(self.tr("status_model_not_found"), 2000)
            except Exception as e:
                print(f"Error loading {self.mode} model setting: {e}")
                if self.status_bar:
                    self.status_bar.showMessage(self.tr("status_error_loading_model"), 5000)

        if loaded_model_key in self.actions:
            self.actions[loaded_model_key].setChecked(True)
        elif self._default_model in self.actions:  # Fallback to default if loaded is somehow invalid
            self.actions[self._default_model].setChecked(True)

    def get_selected_model(self):
        for action in self.actions.values():
            if action.isChecked():
                return action.data()
        return self._default_model  # Fallback, though an action should always be selected

    def get_actions(self):
        return list(self.actions.values())
