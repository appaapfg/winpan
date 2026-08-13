import json
import os

import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from panopticon.data_engine.find_new_detections_coco import (
    find_new_detections_in_coco,
)
from panopticon.data_engine.parse_ds_coco import load_coco_images_and_annotations
from panopticon.draw import draw_annotations_on_image
from panopticon.widgets.model_selector import ModelSelector


class SuggestDetectionsDialog(QDialog):
    """Dialog exposing arguments of find_new_detections_in_coco (excluding device).

    Args to gather:
        model_type: str
        threshold: float
        batch_size: int
        inference_resolution: (H, W)
        iou_threshold: float
        categories: list[str] | 'all'
    """

    def __init__(self, parent, categories=None):
        super().__init__(parent)
        self.setWindowTitle(self.parent().tr("suggest_detections_dialog_title"))
        self.resize(1200, 800)
        self.main_layout = QVBoxLayout(self)

        # Create a scroll area for the main content
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)
        self.scroll_area.setWidget(self.scroll_area_content)
        self.main_layout.addWidget(self.scroll_area)

        # Add a detailed description label at the top
        description_label = QLabel(self.parent().tr("suggest_detections_dialog_description"))
        description_label.setWordWrap(True)
        self.scroll_area_layout.addWidget(description_label)

        self.tab_widget = QTabWidget()
        self.scroll_area_layout.addWidget(self.tab_widget)

        # Create Generate Tab
        self.generate_tab = QWidget()
        self.tab_widget.addTab(self.generate_tab, self.parent().tr("generate_tab"))
        generate_content_layout = QHBoxLayout(self.generate_tab)

        self.generate_left_layout = QVBoxLayout()
        self.generate_right_layout = QVBoxLayout()
        generate_content_layout.addLayout(self.generate_left_layout)
        generate_content_layout.addLayout(self.generate_right_layout)

        # Create Import Tab
        self.import_tab = QWidget()
        self.tab_widget.addTab(self.import_tab, self.parent().tr("import_tab"))
        import_layout = QVBoxLayout(self.import_tab)
        self.import_button = QPushButton(self.parent().tr("import_suggestions_button"))
        self.import_button.clicked.connect(self._import_suggestions)
        import_layout.addWidget(self.import_button)
        self.imported_file_label = QLabel(self.parent().tr("no_file_imported_label"))
        import_layout.addWidget(self.imported_file_label)
        import_layout.addStretch()

        # Shared widgets (categories)
        self.shared_layout = QVBoxLayout()
        self.scroll_area_layout.addLayout(self.shared_layout)

        # Categories (multi-select) - now shared
        self.shared_layout.addWidget(QLabel(self.parent().tr("object_categories_label")))
        self.category_list_widget = QListWidget()
        self.category_list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        if categories:
            for cat_info in categories:
                item = QListWidgetItem(cat_info["name"])
                item.setData(Qt.ItemDataRole.UserRole, cat_info)
                item.setCheckState(Qt.CheckState.Checked)
                self.category_list_widget.addItem(item)
        self.shared_layout.addWidget(self.category_list_widget)

        # Add select all/none buttons for categories
        category_buttons_layout = QHBoxLayout()
        self.select_all_categories_button = QPushButton(self.parent().tr("select_all_button"))
        self.select_all_categories_button.clicked.connect(self._select_all_categories)
        category_buttons_layout.addWidget(self.select_all_categories_button)

        self.select_no_categories_button = QPushButton(self.parent().tr("select_no_button"))
        self.select_no_categories_button.clicked.connect(self._select_no_categories)
        category_buttons_layout.addWidget(self.select_no_categories_button)
        self.shared_layout.addLayout(category_buttons_layout)

        # Model Type selection via ModelSelector for consistency
        self.generate_left_layout.addWidget(QLabel(self.parent().tr("model_type_label")))
        parent_widget = parent if parent is not None else self
        tr_func = getattr(parent_widget, "tr", lambda k, **kw: k)
        status_bar = getattr(parent_widget, "statusBar", lambda: None)()
        self.model_selector = ModelSelector(parent_widget, tr_func, status_bar, auto_load_save=False)
        self.model_type_combo = QComboBox()
        for action in self.model_selector.get_actions():
            model_key = action.data()
            self.model_type_combo.addItem(str(model_key), userData=model_key)
        default_model = self.model_selector.DEFAULT_DETECTION_MODEL
        default_index = self.model_type_combo.findData(default_model)
        if default_index >= 0:
            self.model_type_combo.setCurrentIndex(default_index)
            for act in self.model_selector.get_actions():
                if act.data() == default_model:
                    act.setChecked(True)
                    break

        def _on_combo_changed(index):
            selected_key = self.model_type_combo.itemData(index)
            for act in self.model_selector.get_actions():
                if act.data() == selected_key:
                    act.trigger()
                    break

        self.model_type_combo.currentIndexChanged.connect(_on_combo_changed)
        self.generate_left_layout.addWidget(self.model_type_combo)

        # Threshold
        self.generate_left_layout.addWidget(QLabel(self.parent().tr("threshold_label")))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setDecimals(5)
        self.threshold_input.setValue(0.25)
        self.generate_left_layout.addWidget(self.threshold_input)

        # Batch Size
        self.generate_left_layout.addWidget(QLabel(self.parent().tr("batch_size_label")))
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 512)
        self.batch_size_input.setValue(16)
        self.generate_left_layout.addWidget(self.batch_size_input)

        # Inference width / height (resolution)
        self.generate_left_layout.addWidget(QLabel(self.parent().tr("width_label")))
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 8192)
        self.width_input.setSingleStep(32)
        self.width_input.setValue(640)
        self.generate_left_layout.addWidget(self.width_input)

        self.generate_left_layout.addWidget(QLabel(self.parent().tr("height_label")))
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 8192)
        self.height_input.setSingleStep(32)
        self.height_input.setValue(640)
        self.generate_left_layout.addWidget(self.height_input)

        # IoU Threshold
        self.generate_left_layout.addWidget(QLabel(self.parent().tr("iou_threshold_label")))
        self.iou_threshold_input = QDoubleSpinBox()
        self.iou_threshold_input.setRange(0.0, 1.0)
        self.iou_threshold_input.setSingleStep(0.05)
        self.iou_threshold_input.setValue(0.5)
        self.generate_left_layout.addWidget(self.iou_threshold_input)

        # Mask Type
        self.generate_left_layout.addWidget(QLabel(self.parent().tr("mask_type_label")))
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItem("RLE", userData="RLE")
        self.mask_type_combo.addItem("Polygon", userData="polygon")
        self.generate_left_layout.addWidget(self.mask_type_combo)

        # Reference COCO dataset loader button
        self.load_coco_button = QPushButton(self.parent().tr("load_reference_coco_dataset_button"))
        self.load_coco_button.clicked.connect(self._load_reference_coco_dataset)
        self.generate_right_layout.addWidget(self.load_coco_button)

        # Image pane for showing 1 image
        self.image_pane = QLabel()
        self.image_pane.setFixedSize(640, 480)
        self.image_pane.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_pane.setStyleSheet("border: 1px solid black;")
        self.generate_right_layout.addWidget(self.image_pane)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.parent().tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.parent().tr("cancel_button"))
        self.main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Additional attributes to store the loaded COCO image, detections, and categories
        self.coco_image = None
        self.coco_detections = None
        self.coco_categories = None
        self.imported_suggestions = None

    def _import_suggestions(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle(self.parent().tr("select_suggestions_json_file"))
        file_dialog.setNameFilter(self.parent().tr("json_files_filter"))
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            try:
                with open(selected_file) as f:
                    self.imported_suggestions = json.load(f)
                self.imported_file_label.setText(os.path.basename(selected_file))
            except Exception as e:
                QMessageBox.critical(self, self.parent().tr("error_importing_suggestions"), str(e))
                self.imported_suggestions = None
                self.imported_file_label.setText(self.parent().tr("no_file_imported_label"))

    def _load_reference_coco_dataset(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle(self.parent().tr("select_reference_coco_json_file"))
        file_dialog.setNameFilter(self.parent().tr("json_files_filter"))
        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            try:
                images, annotations_per_image, categories = load_coco_images_and_annotations(selected_file, max_images=1)
                if images:
                    image = images[0]
                    anns = annotations_per_image[0]
                    self.coco_image = image
                    self.coco_detections = anns
                    self.coco_categories = categories
                    annotated_image_np = draw_annotations_on_image(image, anns, categories)
                    h, w, ch = annotated_image_np.shape
                    bytes_per_line = ch * w
                    q_image = QImage(annotated_image_np.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.image_pane.setPixmap(pixmap.scaled(self.image_pane.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                else:
                    self.image_pane.clear()
                    self.image_pane.setText(self.parent().tr("no_images_found"))

                # Update category_list_widget with new categories
                self.category_list_widget.clear()  # Clear existing categories
                for cat_info in categories:
                    item = QListWidgetItem(cat_info["name"])
                    item.setData(Qt.ItemDataRole.UserRole, cat_info)
                    item.setCheckState(Qt.CheckState.Checked)  # Set to checked by default
                    self.category_list_widget.addItem(item)

            except Exception as e:
                self.image_pane.clear()
                self.image_pane.setText(self.parent().tr("error_loading_image") + f": {e}")

    def _select_all_categories(self):
        for i in range(self.category_list_widget.count()):
            item = self.category_list_widget.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def _select_no_categories(self):
        for i in range(self.category_list_widget.count()):
            item = self.category_list_widget.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_categories(self):
        selected_category_names = []
        for i in range(self.category_list_widget.count()):
            item = self.category_list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                cat_info = item.data(Qt.ItemDataRole.UserRole)
                if cat_info:
                    selected_category_names.append(cat_info["name"])
        return selected_category_names

    def generate_suggestions(self, coco_data, coco_json_path, device):
        """
        Run find_new_detections_in_coco with dialog arguments and return suggestions.
        Handles errors internally and shows a message box if needed.
        """
        selected_categories = self._get_selected_categories()
        if not selected_categories:
            QMessageBox.warning(self, self.parent().tr("no_categories_selected_warning", "No Categories Selected"), self.parent().tr("please_select_at_least_one_category", "Please select at least one category."))
            return []

        if self.tab_widget.currentWidget() == self.import_tab:
            if self.imported_suggestions is not None:
                # Filter imported suggestions by selected categories
                # Use coco_data categories to map names to IDs
                current_categories = coco_data.get("categories", [])
                cat_name_to_id = {cat["name"]: cat["id"] for cat in current_categories}
                selected_cat_ids = {cat_name_to_id[name] for name in selected_categories if name in cat_name_to_id}

                filtered_suggestions = [s for s in self.imported_suggestions if s.get("category_id") in selected_cat_ids]
                return filtered_suggestions
            else:
                QMessageBox.warning(self, self.parent().tr("no_suggestions_imported_warning"), self.parent().tr("please_import_a_suggestions_file_first"))
                return []

        try:
            model_type = self.model_selector.get_selected_model()
            if model_type is None:
                raise ValueError("A model must be selected.")
            suggestions = find_new_detections_in_coco(
                coco_input=coco_data,
                images_root=os.path.dirname(coco_json_path),
                model_type=model_type,
                threshold=self.threshold_input.value(),
                batch_size=self.batch_size_input.value(),
                inference_resolution=(self.height_input.value(), self.width_input.value()),
                iou_threshold=self.iou_threshold_input.value(),
                categories=selected_categories,
                box_prompt_referenced_images=self.coco_image,
                box_prompt_detections=self.coco_detections,
                box_prompt_categories=self.coco_categories,
                device=device,
                mask_type=self.mask_type_combo.currentData(),
            )

            if device == "cuda":
                torch.cuda.empty_cache()

            suggestions.sort(key=lambda d: d.get("bbox_confidence", 0), reverse=True)
            return suggestions
        except Exception as e:
            QMessageBox.critical(self, self.parent().tr("error_generating_suggestions"), str(e))
            return []
