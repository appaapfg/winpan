import os

import yaml
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

try:
    from panopticon.config import SETTINGS_FILE
except ImportError:
    from config import SETTINGS_FILE


class ConfidenceThresholdSelectorWidget(QDialog):
    def __init__(self, tr_func, parent=None):
        super().__init__(parent)
        self.tr = tr_func
        self.setWindowTitle(self.tr("confidence_threshold_dialog_title"))
        self.setMinimumWidth(300)

        main_layout = QVBoxLayout(self)

        # Add a detailed description label at the top
        self.description_label = QLabel(self.tr("confidence_threshold_dialog_description"))
        self.description_label.setWordWrap(True)
        main_layout.addWidget(self.description_label)

        # Slider for confidence threshold
        self.slider_label = QLabel(self.tr("confidence_threshold_label"))
        main_layout.addWidget(self.slider_label)

        threshold_layout = QHBoxLayout()
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText(self.tr("confidence_threshold_placeholder"))
        self.text_input.textChanged.connect(self._update_threshold_from_text)
        threshold_layout.addWidget(self.text_input)
        threshold_layout.addWidget(QLabel("%"))
        main_layout.addLayout(threshold_layout)

        # Slider for NMS IOU threshold
        self.iou_slider_label = QLabel(self.tr("nms_iou_threshold_label"))
        main_layout.addWidget(self.iou_slider_label)

        iou_layout = QHBoxLayout()
        self.iou_text_input = QLineEdit(self)
        self.iou_text_input.setPlaceholderText(self.tr("nms_iou_threshold_placeholder"))
        self.iou_text_input.textChanged.connect(self._update_iou_threshold_from_text)
        iou_layout.addWidget(self.iou_text_input)
        iou_layout.addWidget(QLabel("%"))
        main_layout.addLayout(iou_layout)


        # Initialize current_threshold and current_iou_threshold
        self.current_threshold = 0.0
        self.current_iou_threshold = 0.0

        # Buttons
        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton(self.tr("save_button"))
        self.save_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.save_button)

        self.cancel_button = QPushButton(self.tr("cancel_button"))
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(self.button_layout)

        self.load_settings()
        self._update_text_input_from_threshold(self.current_threshold)
        self._update_text_input_from_iou_threshold(self.current_iou_threshold)

    def showEvent(self, event):
        self._previous_threshold_on_show = self.get_confidence_threshold()
        self._previous_iou_threshold_on_show = self.get_nms_iou_threshold()
        super().showEvent(event)

    def _update_text_input_from_threshold(self, value):
        # This method updates the QLineEdit's text based on a numeric threshold
        # To avoid infinite recursion, temporarily block signals
        self.text_input.blockSignals(True)
        self.text_input.setText(f"{value * 100:.0f}")
        self.text_input.blockSignals(False)

    def _update_threshold_from_text(self, text):
        old_threshold = self.current_threshold
        try:
            clean_text = text.replace("%", "")
            if not clean_text:  # Handle empty input
                # If empty, don't change the current_threshold, and revert the text input
                self._update_text_input_from_threshold(old_threshold)
                return

            value = float(clean_text)

            # If value is > 1, assume it's a percentage like "45" and convert to decimal
            if value > 1.0:
                value /= 100.0

            # Check if the value is within the valid range (0.0 < value < 1.0)
            if 0.0 < value < 1.0:
                self.current_threshold = value
            else:
                # If invalid, revert the text input to the last valid value
                self._update_text_input_from_threshold(old_threshold)
        except ValueError:
            # If conversion to float fails, revert the text input
            self._update_text_input_from_threshold(old_threshold)

    def _update_text_input_from_iou_threshold(self, value):
        # This method updates the QLineEdit's text based on a numeric IOU threshold
        # To avoid infinite recursion, temporarily block signals
        self.iou_text_input.blockSignals(True)
        self.iou_text_input.setText(f"{value * 100:.0f}")
        self.iou_text_input.blockSignals(False)

    def _update_iou_threshold_from_text(self, text):
        old_threshold = self.current_iou_threshold
        try:
            clean_text = text.replace("%", "")
            if not clean_text:  # Handle empty input
                # If empty, don't change the current_iou_threshold, and revert the text input
                self._update_text_input_from_iou_threshold(old_threshold)
                return

            value = float(clean_text)

            # If value is > 1, assume it's a percentage like "45" and convert to decimal
            if value > 1.0:
                value /= 100.0

            # Check if the value is within the valid range (0.0 < value < 1.0)
            if 0.0 < value < 1.0:
                self.current_iou_threshold = value
            else:
                # If invalid, revert the text input to the last valid value
                self._update_text_input_from_iou_threshold(old_threshold)
        except ValueError:
            # If conversion to float fails, revert the text input
            self._update_text_input_from_iou_threshold(old_threshold)


    def get_confidence_threshold(self):
        return self.current_threshold

    def get_nms_iou_threshold(self):
        return self.current_iou_threshold


    def load_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    settings = yaml.safe_load(f)
                if settings is None:
                    settings = {}
            except yaml.YAMLError:
                print(f"Error decoding YAML from {SETTINGS_FILE}. Using default settings.")

        # Default to 0.4 if not found or invalid
        threshold = settings.get("confidence_threshold", 0.4)
        self.current_threshold = threshold

        # Default to 0.7 if not found or invalid
        iou_threshold = settings.get("nms_iou_threshold", 0.7)
        self.current_iou_threshold = iou_threshold


        # The text input will be updated by _update_text_input_from_threshold after load_settings is called in __init__
        # and by _update_text_input_from_iou_threshold for IOU threshold
        self._update_text_input_from_threshold(self.current_threshold)
        self._update_text_input_from_iou_threshold(self.current_iou_threshold)

    def save_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    existing_settings = yaml.safe_load(f)
                if existing_settings is not None:
                    settings = existing_settings
            except yaml.YAMLError:
                print(f"Error decoding YAML from {SETTINGS_FILE}. Using default settings.")

        settings["confidence_threshold"] = self.current_threshold
        settings["nms_iou_threshold"] = self.current_iou_threshold

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(settings, f, indent=4)
        except OSError as e:
            print(f"Error saving settings to {SETTINGS_FILE}: {e}")

    def accept(self):
        self.save_settings()
        super().accept()

    def reject(self):
        self.current_threshold = self._previous_threshold_on_show
        self._update_text_input_from_threshold(self.current_threshold)
        self.current_iou_threshold = self._previous_iou_threshold_on_show
        self._update_text_input_from_iou_threshold(self.current_iou_threshold)
        super().reject()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("confidence_threshold_dialog_title"))
        self.description_label.setText(self.tr("confidence_threshold_dialog_description"))
        self.slider_label.setText(self.tr("confidence_threshold_label"))
        self.text_input.setPlaceholderText(self.tr("confidence_threshold_placeholder"))
        self.iou_slider_label.setText(self.tr("nms_iou_threshold_label"))
        self.iou_text_input.setPlaceholderText(self.tr("nms_iou_threshold_placeholder"))
        self.save_button.setText(self.tr("save_button"))
        self.cancel_button.setText(self.tr("cancel_button"))
