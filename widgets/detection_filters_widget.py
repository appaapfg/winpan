import json
import os

import torch
import yaml
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from panopticon.config import CATEGORIES_FILE, SETTINGS_FILE, id2label

# Load categories from JSON and group by supercategory
with open(CATEGORIES_FILE) as f:
    categories_data = json.load(f)
    categories = categories_data.get("categories", [])

# Group categories by supercategory
CATEGORIES_BY_SUPERCATEGORY = {}
FILTER_CATEGORIES = []
for cat in categories:
    supercat = cat["supercategory"]
    name = cat["name"]
    if supercat not in CATEGORIES_BY_SUPERCATEGORY:
        CATEGORIES_BY_SUPERCATEGORY[supercat] = []
    CATEGORIES_BY_SUPERCATEGORY[supercat].append(name)
    FILTER_CATEGORIES.append(name)


class DetectionFiltersWidget(QGroupBox):
    def __init__(self, tr_function, status_bar_instance, parent=None):
        super().__init__(parent)
        self.tr = tr_function
        self._filter_checkboxes = {}
        self._category_labels = {}
        self._category_colors = {}
        self._category_color_buttons = {}
        self._supercat_groups = []

        self.setTitle(self.tr("detection_filters_group_title"))
        main_layout = QVBoxLayout(self)

        # Create a scroll area to hold all supercategory groups
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        for supercategory, category_list in CATEGORIES_BY_SUPERCATEGORY.items():
            # Create a group box for each supercategory
            supercat_group = QGroupBox(self.tr(f"supercategory_{supercategory.lower().replace(' ', '_')}"))
            self._supercat_groups.append(supercat_group)
            supercat_layout = QVBoxLayout(supercat_group)

            for category_key in category_list:
                category_group_layout = QVBoxLayout()

                category_label_text_key = f"category_{category_key.lower().replace(' ', '_')}"
                label = QLabel(f"{self.tr(category_label_text_key)}:")
                self._category_labels[category_key] = label

                color_button = QPushButton(self.tr("color_picker_button"))
                color_button.clicked.connect(lambda _, ck=category_key: self._pick_color(ck))
                self._category_color_buttons[category_key] = color_button
                # Set default color
                default_color = QColor("white")
                self._category_colors[category_key] = default_color
                color_button.setStyleSheet(f"background-color: {default_color.name()};")

                label_layout = QHBoxLayout()
                label_layout.addWidget(label)
                label_layout.addWidget(color_button)
                label_layout.addStretch()

                category_group_layout.addLayout(label_layout)

                checkboxes_layout = QHBoxLayout()
                detect_checkbox = QCheckBox(self.tr("detect_checkbox"))
                notify_checkbox = QCheckBox(self.tr("notify_checkbox"))
                screenshot_checkbox = QCheckBox(self.tr("screenshot_checkbox"))

                self._filter_checkboxes[category_key] = {
                    "detect": detect_checkbox,
                    "notify": notify_checkbox,
                    "screenshot": screenshot_checkbox,
                }

                checkboxes_layout.addWidget(detect_checkbox)
                checkboxes_layout.addWidget(notify_checkbox)
                checkboxes_layout.addWidget(screenshot_checkbox)
                checkboxes_layout.addStretch()

                category_group_layout.addLayout(checkboxes_layout)
                supercat_layout.addLayout(category_group_layout)

            content_layout.addWidget(supercat_group)

        # Add a stretch to push items to the top and avoid cramped layout
        content_layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        self._connect_signals(status_bar_instance)

    def _get_filter_checkboxes(self):
        return self._filter_checkboxes

    def _pick_color(self, category_key):
        current_color = self._category_colors.get(category_key, QColor("white"))
        color = QColorDialog.getColor(current_color, self, f"Pick color for {category_key}")
        if color.isValid():
            self._category_colors[category_key] = color
            button = self._category_color_buttons[category_key]
            button.setStyleSheet(f"background-color: {color.name()};")
            self.save_settings()

    def get_detectable(self) -> list[str]:
        """Returns a list of category keys for which detection is enabled."""
        detectable_categories = []
        for category_key, checkboxes in self._filter_checkboxes.items():
            if checkboxes["detect"].isChecked():
                detectable_categories.append(category_key)
        return detectable_categories

    def get_notifiable(self) -> list[str]:
        """Returns a list of category keys for which notification is enabled."""
        notifiable_categories = []
        for category_key, checkboxes in self._filter_checkboxes.items():
            if checkboxes["notify"].isChecked():
                notifiable_categories.append(category_key)
        return notifiable_categories

    def get_screenshotable(self) -> list[str]:
        """Returns a list of category keys for which auto-screenshot is enabled."""
        screenshotable_categories = []
        for category_key, checkboxes in self._filter_checkboxes.items():
            if checkboxes["screenshot"].isChecked():
                screenshotable_categories.append(category_key)
        return screenshotable_categories

    def get_category_colors(self) -> dict[str, str]:
        """Returns a dictionary mapping category keys to their selected colors as hex strings."""
        category_colors = {}
        for category_key, color in self._category_colors.items():
            category_colors[category_key] = color.name()
        return category_colors

    def convert_ids_to_labels(self, category_key: str | list[str] | torch.Tensor) -> str | list[str]:
        """Converts category IDs to labels using the config mappings."""
        if isinstance(category_key, list):
            # If it's already a list of strings, return as is
            if all(isinstance(item, str) for item in category_key):
                return category_key
            # If it's a list of IDs, convert each
            return [id2label.get(item, f"<Unknown ID: {item}>") for item in category_key]
        elif isinstance(category_key, torch.Tensor):
            if category_key.ndim != 1:
                return []  # Return empty list for non-1D tensors
            labels_from_tensor = []
            for item_tensor_element in category_key:
                try:
                    idx = int(item_tensor_element.item())  # Convert tensor element to Python int
                    label = id2label.get(idx, f"<Unknown ID: {idx}>")
                    labels_from_tensor.append(label)
                except (ValueError, TypeError):
                    # Silently skip items that cannot be converted to int or other processing errors
                    pass
            return labels_from_tensor
        elif isinstance(category_key, str):
            # Return the key directly if it's already a string
            return category_key
        elif isinstance(category_key, int):
            # Convert single ID to label
            return id2label.get(category_key, f"<Unknown ID: {category_key}>")
        else:
            raise TypeError(f"Unsupported type for category_key: {type(category_key)}. Expected str, list, int, or torch.Tensor.")

    def save_settings(self, status_bar_instance=None):
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
                print(f"Error reading settings file for saving filters: {e}")
                if status_bar_instance:
                    status_bar_instance.showMessage(self.tr("status_error_reading_settings", e=e), 5000)
                all_settings = {}

        if not isinstance(all_settings, dict):
            print("Warning: Settings file content was not a dictionary or unreadable. Resetting to empty for this operation.")
            all_settings = {}

        current_filter_settings = {}
        filter_checkboxes = self._get_filter_checkboxes()
        for category, checkboxes in filter_checkboxes.items():
            current_filter_settings[category] = {
                "detect": checkboxes["detect"].isChecked(),
                "notify": checkboxes["notify"].isChecked(),
                "screenshot": checkboxes["screenshot"].isChecked(),
                "color": self._category_colors.get(category, QColor("white")).name(),
            }

        all_settings["detection_filters"] = current_filter_settings

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
            if status_bar_instance:
                status_bar_instance.showMessage(self.tr("status_detection_filter_settings_saved"), 2000)
        except Exception as e:
            print(f"Error writing settings file: {e}")
            if status_bar_instance:
                status_bar_instance.showMessage(self.tr("status_error_writing_settings", e=e), 5000)

    def load_settings(self, status_bar_instance=None):
        if not os.path.exists(SETTINGS_FILE):
            if status_bar_instance:
                status_bar_instance.showMessage(self.tr("status_no_settings_file_filters"), 3000)
            return

        try:
            with open(SETTINGS_FILE) as f:
                all_settings = yaml.safe_load(f)
                if not all_settings or not isinstance(all_settings, dict) or "detection_filters" not in all_settings:
                    if status_bar_instance:
                        status_bar_instance.showMessage(self.tr("status_filters_not_found_or_invalid"), 3000)
                    return

                loaded_filters = all_settings.get("detection_filters", {})
                if not isinstance(loaded_filters, dict):
                    if status_bar_instance:
                        status_bar_instance.showMessage(self.tr("status_invalid_filter_format"), 3000)
                    return

                filter_checkboxes = self._get_filter_checkboxes()
                for category, states in loaded_filters.items():
                    if category in filter_checkboxes and isinstance(states, dict):
                        checkboxes = filter_checkboxes[category]
                        detect_cb = checkboxes.get("detect")
                        notify_cb = checkboxes.get("notify")
                        screenshot_cb = checkboxes.get("screenshot")

                        if detect_cb and "detect" in states:
                            detect_cb.blockSignals(True)
                            detect_cb.setChecked(bool(states["detect"]))
                            detect_cb.blockSignals(False)
                        if notify_cb and "notify" in states:
                            notify_cb.blockSignals(True)
                            notify_cb.setChecked(bool(states["notify"]))
                            notify_cb.blockSignals(False)
                        if screenshot_cb and "screenshot" in states:
                            screenshot_cb.blockSignals(True)
                            screenshot_cb.setChecked(bool(states["screenshot"]))
                            screenshot_cb.blockSignals(False)
                        if "color" in states:
                            color_str = states["color"]
                            try:
                                color = QColor(color_str)
                                self._category_colors[category] = color
                                button = self._category_color_buttons.get(category)
                                if button:
                                    button.setStyleSheet(f"background-color: {color.name()};")
                            except Exception:
                                pass  # ignore invalid color
                if status_bar_instance:
                    status_bar_instance.showMessage(self.tr("status_filter_settings_loaded"), 2000)
        
        except Exception as e:
            print(f"Error loading detection filter settings: {e}")
            if status_bar_instance:
                status_bar_instance.showMessage(self.tr("status_error_loading_settings", e=e), 5000)

    def update_available_classes(self, supported_names: list[str] | None):
        """Enable/disable category checkboxes based on supported model classes.

        Args:
            supported_names: list of supported category names, or None to indicate
                the model supports arbitrary text prompts (enable all categories).
        """
        # If supported_names is None, enable all checkboxes
        for category_key, checkboxes in self._filter_checkboxes.items():
            detect_cb = checkboxes.get("detect")
            notify_cb = checkboxes.get("notify")
            screenshot_cb = checkboxes.get("screenshot")

            if supported_names is None:
                for cb in (detect_cb, notify_cb, screenshot_cb):
                    if cb is not None:
                        cb.setEnabled(True)
                        cb.setToolTip("")
            else:
                if category_key in supported_names:
                    for cb in (detect_cb, notify_cb, screenshot_cb):
                        if cb is not None:
                            cb.setEnabled(True)
                            cb.setToolTip("")
                else:
                    # Disable and clear selection for unsupported categories
                    for cb in (detect_cb, notify_cb, screenshot_cb):
                        if cb is not None:
                            cb.blockSignals(True)
                            cb.setChecked(False)
                            cb.blockSignals(False)
                            cb.setEnabled(False)
                            try:
                                cb.setToolTip(self.tr("status_class_not_supported"))
                            except Exception:
                                cb.setToolTip("Not supported by current model")

        # Handled earlier within load_settings; no-op here.
        # make sure scrollbars stay active (the scroll area might otherwise gray out)
        for sb in self.findChildren(QScrollBar):
            sb.setEnabled(True)

    def retranslate_ui(self):
        self.setTitle(self.tr("detection_filters_group_title"))
        for i, supercategory in enumerate(CATEGORIES_BY_SUPERCATEGORY.keys()):
            self._supercat_groups[i].setTitle(self.tr(f"supercategory_{supercategory.lower().replace(' ', '_')}"))
        for category_key in FILTER_CATEGORIES:
            category_label_text_key = f"category_{category_key.lower().replace(' ', '_')}"
            if category_key in self._category_labels:
                self._category_labels[category_key].setText(f"{self.tr(category_label_text_key)}:")

            if category_key in self._filter_checkboxes:
                checkboxes = self._filter_checkboxes[category_key]
                checkboxes["detect"].setText(self.tr("detect_checkbox"))
                checkboxes["notify"].setText(self.tr("notify_checkbox"))
                checkboxes["screenshot"].setText(self.tr("screenshot_checkbox"))

            if category_key in self._category_color_buttons:
                self._category_color_buttons[category_key].setText(self.tr("color_picker_button"))

    def _save_settings_on_change(self, state, status_bar_instance):
        """Saves settings when a checkbox state changes."""
        self.save_settings(status_bar_instance)

    def _connect_signals(self, status_bar_instance):
        """Connects stateChanged signals of all checkboxes to the save_settings method."""
        for category_key in FILTER_CATEGORIES:
            if category_key in self._filter_checkboxes:
                checkboxes = self._filter_checkboxes[category_key]
                checkboxes["detect"].stateChanged.connect(lambda state, sbi=status_bar_instance: self._save_settings_on_change(state, sbi))
                checkboxes["notify"].stateChanged.connect(lambda state, sbi=status_bar_instance: self._save_settings_on_change(state, sbi))
                checkboxes["screenshot"].stateChanged.connect(lambda state, sbi=status_bar_instance: self._save_settings_on_change(state, sbi))
