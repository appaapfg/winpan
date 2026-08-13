import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCursor,
    QIcon,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from panopticon.data_engine.add_bboxes_from_segmentation import (
    add_bboxes_from_segmentation,
)
from panopticon.data_engine.convert_segmentation_coco import convert_coco_mask_type
from panopticon.data_engine.remap_ids import (
    _load_category_mapping,
    remap_category_ids,
)
from panopticon.data_engine.remove_segmentation_masks_coco import (
    remove_segmentation_masks_from_coco,
)
from panopticon.data_engine.remove_unannotated_images_coco import (
    remove_unannotated_images_coco,
)
from panopticon.data_engine.split_coco import split_and_save_coco
from panopticon.draw import (
    draw_annotations_on_image,
    draw_dashed_bbox_with_label,
    draw_minus,
    draw_plus,
    draw_segmentation_mask,
    overlay_candidate_segmentations,
)
from panopticon.mode_controllers import (
    AddAIModeController,
    AddAIMultiPointModeController,
    AddManualModeController,
    ChangeAnnotationModeController,
    EditBoundingBoxModeController,
    RemoveAnnotationModeController,
)
from panopticon.my_inference.segment import Segmenter
from panopticon.translations_data import TRANSLATIONS
from panopticon.config import VERSION
from panopticon.widgets.about_panannotator_dialog import AboutPanAnnotatorDialog
from panopticon.widgets.add_bboxes_from_segmentation_dialog import (
    AddBboxesFromSegmentationDialog,
)
from panopticon.widgets.bbox_thickness_selector_widget import (
    BBoxThicknessSelectorWidget,
)
from panopticon.widgets.class_selector import ClassSelector
from panopticon.widgets.create_unlabeled_coco_dialog import CreateUnlabeledCocoDialog
from panopticon.widgets.dataset_info_dialog import DatasetInfoDialog
from panopticon.widgets.detection_filters_widget import DetectionFiltersWidget
from panopticon.widgets.device_selector import DeviceSelector
from panopticon.widgets.icon_utils import get_zoom_icon
from panopticon.widgets.image_info_pane import ImageInfoPane
from panopticon.widgets.image_selector_widget import ImageSelectorWidget
from panopticon.widgets.inference_resolution_selector_widget import (
    InferenceResolutionSelectorWidget,
)
from panopticon.widgets.language_selector import (
    LanguageActionGroup as LanguageActionGroupCustom,
)
from panopticon.widgets.model_selector import ModelSelector
from panopticon.widgets.remap_ids_dialog import RemapIDsDialog
from panopticon.widgets.remove_categories_dialog import RemoveCategoriesDialog
from panopticon.widgets.segment_bboxes_dialog import SegmentBBoxesDialog
from panopticon.widgets.segmentation_opacity_selector_widget import (
    SegmentationOpacitySelectorWidget,
)
from panopticon.widgets.split_coco_dialog import SplitCocoDialog
from panopticon.widgets.suggest_detections_dialog import SuggestDetectionsDialog
from panopticon.widgets.suggestion_control_widget import SuggestionControlWidget
from panopticon.widgets.theme_manager import ThemeActionGroup
from panopticon.widgets.video_source_selector_widget import VideoSourceSelectorWidget

MIN_MANUAL_BOX_SIZE = 8
DEFAULT_MODE = "view"


class PanAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_language = "en"
        self.setWindowTitle(self.tr("pan_annotator_title", version=VERSION))

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # COCO Data
        self.coco_data = {"categories": [], "annotations": [], "images": []}
        self.unsaved_changes = False

        # Left Pane - Class Selector
        self.class_selector = ClassSelector(self.coco_data["categories"], self.tr, TRANSLATIONS)
        # When a new custom category is added via the selector, append it to coco_data
        self.class_selector.custom_category_added.connect(self._on_custom_category_added)
        self.class_selector.setFixedWidth(200)
        main_layout.addWidget(self.class_selector)

        # Right Pane - Image Viewer
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setScaledContents(False)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Enable Ctrl + Mouse Wheel zoomon the image viewport
        self.scroll_area.viewport().installEventFilter(self)

        # Create a vertical layout for the image viewer and image selector
        right_pane_layout = QVBoxLayout()
        right_pane_layout.addWidget(self.scroll_area)

        # Image Selector Widget
        self.image_selector_widget = ImageSelectorWidget(self, self.tr)
        self.image_selector_widget.image_selected.connect(self._on_image_selected)
        self.image_selector_widget.setMaximumHeight(150)
        right_pane_layout.addWidget(self.image_selector_widget)

        main_layout.addLayout(right_pane_layout)
        self._right_pane_layout = right_pane_layout  # store for widget swapping

        # Image Info Pane
        self.image_info_pane = ImageInfoPane(self.tr)
        main_layout.addWidget(self.image_info_pane)

        self.image = None
        self.zoom_scale = 1.0
        self.zoom_on_suggestion_level = 1
        self.mode = DEFAULT_MODE  # 'view', 'add', 'add-AI', 'remove', 'edit', 'change', 'suggest'

        self.coco_json_path = None
        self.current_image_index = -1
        self.current_image_id = -1

        # Suggestion mode state
        self.suggestion_mode_active = False
        self.suggestions = []  # list of dicts
        self.current_suggestion_index = -1
        self.current_suggestion = None
        self.suggestion_control_widget = None
        self._suggestion_editing_handle = None
        self._suggestion_drag_start_pos = None
        self._suggestion_original_bbox = None

        self.interpolation_mode = "nearest"

        self.add_manual_controller = AddManualModeController(self)
        self.add_ai_controller = AddAIModeController(self)
        self.add_ai_multi_point_controller = AddAIMultiPointModeController(self)
        self.remove_annotation_controller = RemoveAnnotationModeController(self)
        self.edit_bbox_controller = EditBoundingBoxModeController(self)
        self.change_annotation_controller = ChangeAnnotationModeController(self)

        self._create_actions()
        self.suggest_mode_action.setEnabled(False)
        self._create_menus()
        self._create_toolbar()

        self.segmenter = Segmenter(model_type=self.segmentation_model_selector.get_selected_model(), device=self.device_selector.get_selected_device())
        self.theme_manager = ThemeActionGroup(self.configuration_menu, self, QApplication.instance(), self.tr, self.statusBar())

        self.language_manager = LanguageActionGroupCustom(self, self.tr, self.set_language, self.statusBar())
        self.configuration_menu.addMenu(self.language_manager)

        temp_detection_filters_widget = DetectionFiltersWidget(self.tr, None, parent=None)
        temp_detection_filters_widget.load_settings(None)
        self.category_colors = temp_detection_filters_widget.get_category_colors()

        self.retranslate_ui()

    def _on_custom_category_added(self, category_dict):
        # Avoid duplicates by ID
        self.coco_data["categories"].append(
            {
                "id": category_dict.get("id"),
                "name": category_dict.get("name"),
                "supercategory": category_dict.get("supercategory", ""),
            }
        )
        self.unsaved_changes = True
        # Optionally update any UI pieces that rely on categories
        # (ClassSelector already shows it; if other widgets depend, refresh here.)

    def _on_image_selected(self, image_path, image_obj, annotations, image_id):
        # Store received data from ImageSelectorWidget signal (path, image object, annotations, image id)
        self.current_image_path = image_path
        self.image = image_obj
        self.current_image_id = image_id  # Store the current image ID
        self.fit_to_view()
        self._display_image_with_annotations()

        # Update image info pane
        if self.image:
            name = os.path.basename(image_path) if image_path else ""
            img_id = str(image_id)
            self.image_info_pane.set_image_info(name, img_id, self.image)
        else:
            self.image_info_pane.clear()

    def load_coco_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("load_coco_dataset_action"), "", self.tr("json_files_filter"))
        if file_path:
            if self.unsaved_changes:
                reply = QMessageBox.question(
                    self,
                    self.tr("unsaved_changes_title"),
                    self.tr("unsaved_changes_message"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._save_annotations()
                elif reply == QMessageBox.StandardButton.Cancel:
                    self.statusBar().showMessage(self.tr("status_coco_load_cancelled"), 2000)
                    return
            self.load_coco_dataset_from_path(file_path)
        else:
            self.statusBar().showMessage(self.tr("status_coco_load_cancelled"), 2000)

    def load_coco_dataset_from_path(self, file_path):
        try:
            with open(file_path) as f:
                self.coco_data = json.load(f)
                self.coco_json_path = file_path
                self.unsaved_changes = False
                self.image_selector_widget.set_coco_data(self.coco_data, self.coco_json_path)

                # Populate ClassSelector with a list of category dicts
                categories_list = self.coco_data.get("categories", [])
                self.class_selector.populate_categories(categories_list)

                self.statusBar().showMessage(
                    self.tr("status_coco_dataset_loaded", file_path=file_path),
                    3000,
                )
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("error_loading_coco_title"),
                self.tr(
                    "error_loading_coco_message",
                    file_path=file_path,
                    e=e,
                ),
            )

    def set_language(self, language_code):
        self.current_language = language_code
        self.language_manager.save_language_setting(language_code)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("pan_annotator_title", version=VERSION))
        self.save_annotations_action.setText(self.tr("save_annotations_action"))
        self.save_annotations_as_action.setText(self.tr("save_annotations_as_action"))
        self.load_coco_dataset_action.setText(self.tr("load_coco_dataset_action"))
        self.create_unlabeled_coco_action.setText(self.tr("create_unlabeled_coco_action"))
        self.exit_action.setText(self.tr("exit_action"))
        self.split_dataset_action.setText(self.tr("split_dataset_action"))
        self.merge_datasets_action.setText(self.tr("merge_datasets_action"))
        self.zoom_in_action.setText(self.tr("zoom_in_action"))
        self.zoom_out_action.setText(self.tr("zoom_out_action"))
        self.fit_to_view_action.setText(self.tr("fit_to_view_action"))
        self.view_mode_action.setText(self.tr("view_mode_action"))
        self.add_ai_mode_action.setText(self.tr("add_ai_annotation_action"))
        self.remove_mode_action.setText(self.tr("remove_annotation_action"))
        self.edit_mode_action.setText(self.tr("edit_bounding_box_action"))
        self.change_mode_action.setText(self.tr("change_annotation_action"))
        self.suggest_mode_action.setText(self.tr("suggest_mode_action"))
        self.add_manual_mode_action.setText(self.tr("add_manual_annotation_action"))
        self.add_ai_multi_point_mode_action.setText(self.tr("add_ai_multi_point_annotation_action"))
        self.suggest_detections_action.setText(self.tr("suggest_detections_action"))
        self.show_annotations_action.setText(self.tr("show_annotations_action"))
        self.file_menu.setTitle(self.tr("file_menu"))
        self.configuration_menu.setTitle(self.tr("configure_menu"))
        self.segmentation_model_menu.setTitle(self.tr("select_segmentation_model_menu"))
        self.edit_menu.setTitle(self.tr("edit_menu"))
        self.main_toolbar.setWindowTitle(self.tr("main_toolbar"))
        self.edit_toolbar.setWindowTitle(self.tr("edit_toolbar"))
        self.theme_manager.retranslate_ui()
        self.language_manager.retranslate_ui()
        self.class_selector.custom_id_button.setText(self.tr("custom_object_id_button"))
        self.image_selector_widget.retranslate_ui()
        self.class_selector.retranslate_ui()
        self.segmentation_opacity_selector_widget.retranslate_ui()
        self.bbox_thickness_selector_widget.retranslate_ui()
        self.draw_labels_action.setText(self.tr("draw_labels_action"))
        self.bbox_thickness_action.setText(self.tr("bbox_thickness_action"))
        self.segment_bboxes_action.setText(self.tr("segment_existing_bboxes_action"))
        self.remove_categories_action.setText(self.tr("remove_categories_action"))
        self.remap_ids_action.setText(self.tr("remap_ids_action"))
        self.remove_current_image_action.setText(self.tr("remove_current_image_action"))
        self.remove_segmentation_masks_action.setText(self.tr("remove_segmentation_masks_action"))
        self.remove_unannotated_images_action.setText(self.tr("remove_unannotated_images_action"))
        self.segmentation_opacity_action.setText(self.tr("segmentation_opacity_action"))
        self.segmentation_inference_resolution_action.setText(self.tr("configure_inference_resolution_action"))
        self.video_sources_action.setText(self.tr("configure_video_sources_action"))
        self.help_menu.setTitle(self.tr("help_menu"))
        self.about_panannotator_action.setText(self.tr("about_panannotator_action"))
        self.dataset_info_action.setText(self.tr("dataset_info_action"))
        self.device_selector.retranslate_ui()
        self.convert_segmentation_menu.setTitle(self.tr("convert_segmentation_mask_menu"))
        self.convert_to_rle_action.setText(self.tr("convert_to_rle_action"))
        self.convert_to_polygons_action.setText(self.tr("convert_to_polygons_action"))
        self.add_bboxes_from_segmentation_action.setText(self.tr("add_bboxes_from_segmentation_action"))
        self.add_bboxes_current_overwrite_action.setText(self.tr("add_bboxes_current_overwrite_action"))
        self.zoom_on_suggestion_menu.setTitle(self.tr("zoom_on_suggestion_menu"))
        self.zoom_disable_action.setText(self.tr("zoom_disable_action"))
        self.zoom_2x_action.setText("2x")
        self.zoom_3x_action.setText("3x")
        self.zoom_4x_action.setText("4x")
        self.interpolation_menu.setTitle(self.tr("interpolation_mode_menu"))
        self.nearest_action.setText(self.tr("nearest_neighbor_action"))
        self.smooth_action.setText(self.tr("bilinear_action"))
        self.image_info_pane.retranslate_ui()
        if self.suggestion_control_widget is not None:
            self.suggestion_control_widget.retranslate_ui()

    def tr(self, text: str, **kwargs) -> str:
        if self.current_language in TRANSLATIONS:
            translation = TRANSLATIONS[self.current_language].get(text, text)
        else:
            translation = TRANSLATIONS["en"].get(text, text)

        if kwargs:
            try:
                return translation.format(**kwargs)
            except KeyError as e:
                print(f"Translation key error: {e} for text: '{text}'")
                return text
        return translation

    def _create_actions(self):
        self.save_annotations_action = QAction(self.tr("save_annotations_action"), self)
        self.save_annotations_action.triggered.connect(self._save_annotations)
        self.save_annotations_action.setShortcut("Ctrl+S")

        self.save_annotations_as_action = QAction(self.tr("save_annotations_as_action"), self)
        self.save_annotations_as_action.triggered.connect(self._save_annotations_as)
        self.save_annotations_as_action.setShortcut("Ctrl+Shift+S")

        self.load_coco_dataset_action = QAction(self.tr("load_coco_dataset_action"), self)
        self.load_coco_dataset_action.triggered.connect(self.load_coco_dataset)
        self.load_coco_dataset_action.setShortcut("Ctrl+O")
        self.create_unlabeled_coco_action = QAction(self.tr("create_unlabeled_coco_action"), self)
        self.create_unlabeled_coco_action.triggered.connect(self._create_unlabeled_coco_dataset)
        self.create_unlabeled_coco_action.setShortcut("Ctrl+N")

        self.exit_action = QAction(self.tr("exit_action"), self)
        self.exit_action.triggered.connect(self.close)
        self.exit_action.setShortcut("Ctrl+Q")

        self.split_dataset_action = QAction(self.tr("split_dataset_action"), self)
        self.split_dataset_action.triggered.connect(self._split_dataset)

        self.merge_datasets_action = QAction(self.tr("merge_datasets_action"), self)
        self.merge_datasets_action.triggered.connect(self._merge_datasets)

        self.zoom_in_action = QAction(self.tr("zoom_in_action"), self)
        self.zoom_in_action.triggered.connect(lambda: self._adjust_zoom(1.2))
        self.zoom_in_action.setIcon(get_zoom_icon("zoom-in", "plus", self.palette()))

        self.zoom_out_action = QAction(self.tr("zoom_out_action"), self)
        self.zoom_out_action.triggered.connect(lambda: self._adjust_zoom(1 / 1.2))
        self.zoom_out_action.setIcon(get_zoom_icon("zoom-out", "minus", self.palette()))

        self.fit_to_view_action = QAction(self.tr("fit_to_view_action"), self)
        self.fit_to_view_action.triggered.connect(self.fit_to_view)
        self.fit_to_view_action.setShortcut("Ctrl+0")
        self.fit_to_view_action.setIcon(get_zoom_icon("zoom-fit-best", "fit", self.palette()))

        self.view_mode_action = QAction(self.tr("view_mode_action"), self)
        self.view_mode_action.setCheckable(True)
        self.view_mode_action.setChecked(True)
        self.view_mode_action.triggered.connect(self._set_view_mode)
        self.view_mode_action.setShortcut("F1")

        self.add_manual_mode_action = QAction(self.tr("add_manual_annotation_action"), self)
        self.add_manual_mode_action.setCheckable(True)
        self.add_manual_mode_action.triggered.connect(self._set_add_manual_mode)
        self.add_manual_mode_action.setShortcut("F2")

        self.add_ai_mode_action = QAction(self.tr("add_ai_annotation_action"), self)
        self.add_ai_mode_action.setCheckable(True)
        self.add_ai_mode_action.triggered.connect(self._set_add_ai_mode)
        self.add_ai_mode_action.setShortcut("F3")

        self.add_ai_multi_point_mode_action = QAction(self.tr("add_ai_multi_point_annotation_action"), self)
        self.add_ai_multi_point_mode_action.setCheckable(True)
        self.add_ai_multi_point_mode_action.triggered.connect(self._set_add_ai_multi_point_mode)
        self.add_ai_multi_point_mode_action.setShortcut("F4")

        self.remove_mode_action = QAction(self.tr("remove_annotation_action"), self)
        self.remove_mode_action.setCheckable(True)
        self.remove_mode_action.triggered.connect(self._set_remove_mode)
        self.remove_mode_action.setShortcut("F5")

        self.edit_mode_action = QAction(self.tr("edit_bounding_box_action"), self)
        self.edit_mode_action.setCheckable(True)
        self.edit_mode_action.triggered.connect(self._set_edit_mode)
        self.edit_mode_action.setShortcut("F6")

        self.change_mode_action = QAction(self.tr("change_annotation_action"), self)
        self.change_mode_action.setCheckable(True)
        self.change_mode_action.triggered.connect(self._set_change_mode)
        self.change_mode_action.setShortcut("F7")

        self.suggest_mode_action = QAction(self.tr("suggest_mode_action"), self)
        self.suggest_mode_action.setCheckable(True)
        self.suggest_mode_action.triggered.connect(self._set_suggest_mode)
        self.suggest_mode_action.setShortcut("F8")

        self.suggest_detections_action = QAction(self.tr("suggest_detections_action"), self)
        self.suggest_detections_action.triggered.connect(self._start_suggestion_mode)
        self.segment_bboxes_action = QAction(self.tr("segment_existing_bboxes_action"), self)
        self.segment_bboxes_action.triggered.connect(self._segment_existing_bboxes)

        self.remove_categories_action = QAction(self.tr("remove_categories_action"), self)
        self.remove_categories_action.triggered.connect(self._remove_categories)
        self.remap_ids_action = QAction(self.tr("remap_ids_action"), self)
        self.remap_ids_action.triggered.connect(self._remap_category_ids)

        self.remove_current_image_action = QAction(self.tr("remove_current_image_action"), self)
        self.remove_current_image_action.triggered.connect(self._remove_current_image)
        self.remove_current_image_action.setShortcut("Ctrl+Del")

        self.dataset_info_action = QAction(self.tr("dataset_info_action"), self)
        self.dataset_info_action.triggered.connect(self._show_dataset_info)

        self.remove_unannotated_images_action = QAction(self.tr("remove_unannotated_images_action"), self)
        self.remove_unannotated_images_action.triggered.connect(self._remove_unannotated_images)

        self.convert_to_rle_action = QAction(self.tr("convert_to_rle_action"), self)
        self.convert_to_rle_action.triggered.connect(self._convert_to_rle)

        self.convert_to_polygons_action = QAction(self.tr("convert_to_polygons_action"), self)
        self.convert_to_polygons_action.triggered.connect(self._convert_to_polygons)

        self.remove_segmentation_masks_action = QAction(self.tr("remove_segmentation_masks_action"), self)
        self.remove_segmentation_masks_action.triggered.connect(self._remove_segmentation_masks)

        self.add_bboxes_from_segmentation_action = QAction(self.tr("add_bboxes_from_segmentation_action"), self)
        self.add_bboxes_from_segmentation_action.triggered.connect(self._add_bboxes_from_segmentation)

        self.add_bboxes_current_overwrite_action = QAction(
            self.tr("add_bboxes_current_overwrite_action"),
            self,
        )
        self.add_bboxes_current_overwrite_action.setShortcut("Ctrl+Alt+B")
        self.add_bboxes_current_overwrite_action.triggered.connect(self._add_bboxes_from_segmentation_current_overwrite)

        self.change_categories_from_file_action = QAction(self.tr("import_categories_from_file_action"), self)
        self.change_categories_from_file_action.triggered.connect(self._change_categories_from_file)

        self.show_annotations_action = QAction(self.tr("show_annotations_action"), self)
        self.show_annotations_action.setCheckable(True)
        self.show_annotations_action.setChecked(True)
        self.show_annotations_action.triggered.connect(lambda: self._display_image_with_annotations())
        self.show_annotations_action.setShortcut("Ctrl+1")

        self.draw_labels_action = QAction(self.tr("draw_labels_action"), self)
        self.draw_labels_action.setCheckable(True)
        self.draw_labels_action.setChecked(True)
        self.draw_labels_action.triggered.connect(lambda: self._display_image_with_annotations())
        self.draw_labels_action.setShortcut("Ctrl+2")

    def _create_menus(self):
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu(self.tr("file_menu"))
        self.file_menu.addAction(self.save_annotations_action)
        self.file_menu.addAction(self.save_annotations_as_action)
        self.file_menu.addAction(self.create_unlabeled_coco_action)
        self.file_menu.addAction(self.load_coco_dataset_action)
        self.file_menu.addAction(self.split_dataset_action)
        self.file_menu.addAction(self.merge_datasets_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.dataset_info_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = menu_bar.addMenu(self.tr("edit_menu"))
        self.edit_menu.addAction(self.suggest_detections_action)
        self.edit_menu.addAction(self.segment_bboxes_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.remove_categories_action)
        self.edit_menu.addAction(self.remap_ids_action)
        self.edit_menu.addAction(self.change_categories_from_file_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.remove_current_image_action)
        self.edit_menu.addAction(self.remove_unannotated_images_action)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.add_bboxes_from_segmentation_action)
        self.edit_menu.addAction(self.add_bboxes_current_overwrite_action)
        self.edit_menu.addSeparator()
        self.convert_segmentation_menu = self.edit_menu.addMenu(self.tr("convert_segmentation_mask_menu"))
        self.convert_segmentation_menu.addAction(self.convert_to_rle_action)
        self.convert_segmentation_menu.addAction(self.convert_to_polygons_action)
        self.edit_menu.addAction(self.remove_segmentation_masks_action)

        self.configuration_menu = menu_bar.addMenu(self.tr("configure_menu"))
        self.segmentation_model_menu = self.configuration_menu.addMenu(self.tr("select_segmentation_model_menu"))
        self.segmentation_model_selector = ModelSelector(self, self.tr, self.statusBar(), mode="segmentation")
        self.segmentation_model_menu.addActions(self.segmentation_model_selector.get_actions())
        self.segmentation_model_selector.triggered.connect(self._on_segmentation_model_switched)
        self.segmentation_opacity_selector_widget = SegmentationOpacitySelectorWidget(self.tr, self)
        self.segmentation_opacity_action = QAction(self.tr("segmentation_opacity_action"), self)
        self.segmentation_opacity_action.triggered.connect(self._open_segmentation_opacity_dialog)
        self.configuration_menu.addAction(self.segmentation_opacity_action)
        self.bbox_thickness_selector_widget = BBoxThicknessSelectorWidget(self.tr, self)
        self.bbox_thickness_action = QAction(self.tr("bbox_thickness_action"), self)
        self.bbox_thickness_action.triggered.connect(self._open_bbox_thickness_dialog)
        self.configuration_menu.addAction(self.bbox_thickness_action)
        self.segmentation_inference_resolution_widget = InferenceResolutionSelectorWidget(self.tr, self, config_key_name="segmentation_inference_resolution")
        self.segmentation_inference_resolution_action = QAction(self.tr("configure_inference_resolution_action"), self)
        self.segmentation_inference_resolution_action.triggered.connect(lambda: self.segmentation_inference_resolution_widget.exec())
        self.configuration_menu.addAction(self.segmentation_inference_resolution_action)
        self.video_sources_widget = VideoSourceSelectorWidget(self.tr, self)
        self.video_sources_action = QAction(self.tr("configure_video_sources_action"), self)
        self.video_sources_action.triggered.connect(lambda: self.video_sources_widget.exec())
        self.configuration_menu.addAction(self.video_sources_action)
        self.device_selector = DeviceSelector(self, self.tr, self.statusBar(), settings_key="pan_annotator_device")
        self.device_selector.device_changed.connect(self._on_device_switched)
        self.configuration_menu.addMenu(self.device_selector)
        self.configuration_menu.addAction(self.show_annotations_action)
        self.configuration_menu.addAction(self.draw_labels_action)
        self.zoom_on_suggestion_menu = self.configuration_menu.addMenu(self.tr("zoom_on_suggestion_menu"))
        self.zoom_disable_action = QAction(self.tr("zoom_disable_action"), self)
        self.zoom_disable_action.setCheckable(True)
        self.zoom_disable_action.setChecked(True)
        self.zoom_2x_action = QAction("2x", self)
        self.zoom_2x_action.setCheckable(True)
        self.zoom_3x_action = QAction("3x", self)
        self.zoom_3x_action.setCheckable(True)
        self.zoom_4x_action = QAction("4x", self)
        self.zoom_4x_action.setCheckable(True)
        self.zoom_on_suggestion_menu.addAction(self.zoom_disable_action)
        self.zoom_on_suggestion_menu.addAction(self.zoom_2x_action)
        self.zoom_on_suggestion_menu.addAction(self.zoom_3x_action)
        self.zoom_on_suggestion_menu.addAction(self.zoom_4x_action)
        self.zoom_group = QActionGroup(self)
        self.zoom_group.addAction(self.zoom_disable_action)
        self.zoom_group.addAction(self.zoom_2x_action)
        self.zoom_group.addAction(self.zoom_3x_action)
        self.zoom_group.addAction(self.zoom_4x_action)
        self.zoom_group.setExclusive(True)
        self.zoom_disable_action.triggered.connect(lambda: setattr(self, "zoom_on_suggestion_level", 1))
        self.zoom_2x_action.triggered.connect(lambda: setattr(self, "zoom_on_suggestion_level", 2))
        self.zoom_3x_action.triggered.connect(lambda: setattr(self, "zoom_on_suggestion_level", 3))
        self.zoom_4x_action.triggered.connect(lambda: setattr(self, "zoom_on_suggestion_level", 4))

        self.interpolation_menu = self.configuration_menu.addMenu(self.tr("interpolation_mode_menu"))
        self.nearest_action = QAction(self.tr("nearest_neighbor_action"), self)
        self.nearest_action.setCheckable(True)
        self.nearest_action.setChecked(True)
        self.smooth_action = QAction(self.tr("bilinear_action"), self)
        self.smooth_action.setCheckable(True)
        self.interpolation_menu.addAction(self.nearest_action)
        self.interpolation_menu.addAction(self.smooth_action)
        self.interpolation_group = QActionGroup(self)
        self.interpolation_group.addAction(self.nearest_action)
        self.interpolation_group.addAction(self.smooth_action)
        self.interpolation_group.setExclusive(True)
        self.nearest_action.triggered.connect(lambda: (setattr(self, "interpolation_mode", "nearest"), self._display_image_with_annotations()))
        self.smooth_action.triggered.connect(lambda: (setattr(self, "interpolation_mode", "smooth"), self._display_image_with_annotations()))

        # Help menu integration
        self.help_menu = menu_bar.addMenu(self.tr("help_menu"))
        self.about_panannotator_action = QAction(self.tr("about_panannotator_action"), self)
        self.about_panannotator_action.triggered.connect(lambda: AboutPanAnnotatorDialog(self.tr, self).exec())
        self.help_menu.addAction(self.about_panannotator_action)

    def _remove_categories(self):
        # Open dialog to remove categories; no translations per request
        categories = self.coco_data.get("categories", [])
        if not categories:
            QMessageBox.information(self, self.tr("info_title"), self.tr("info_no_categories_to_remove"))
            return
        dlg = RemoveCategoriesDialog(self, categories=categories)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        result = dlg.get_selection()
        ids_to_remove = set(result.get("category_ids", []))
        if not ids_to_remove:
            return
        # Remove categories
        self.coco_data["categories"] = [c for c in categories if c.get("id") not in ids_to_remove]
        # Optionally remove annotations for those categories
        if result.get("remove_annotations", True):
            anns = self.coco_data.get("annotations", [])
            self.coco_data["annotations"] = [a for a in anns if a.get("category_id") not in ids_to_remove]
        self.unsaved_changes = True
        # Refresh class selector and clear selection
        try:
            self.class_selector.populate_categories(self.coco_data["categories"])
            self.class_selector._unselect_current()
        except Exception:
            pass
        # Redraw view in case any boxes disappeared
        self._display_image_with_annotations()

    def _remap_category_ids(self):
        # Dialog to remap IDs
        dlg = RemapIDsDialog(self.tr, self, categories=self.coco_data.get("categories", []))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        values = dlg.get_values()
        mapping_text = values.get("category_mapping", "").strip()
        if not mapping_text:
            return
        allow_collisions = bool(values.get("allow_collisions"))
        try:
            category_mapping = _load_category_mapping(mapping_text)
        except Exception as e:
            QMessageBox.critical(self, self.tr("error_title"), self.tr("error_failed_to_parse_mapping", e=e))
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(self.coco_data, tf)
                tmp_path = tf.name
            new_coco = remap_category_ids(tmp_path, category_mapping, allow_collisions=allow_collisions, output_file=None)
            self.coco_data["categories"] = new_coco.dataset.get("categories", [])
            self.coco_data["annotations"] = new_coco.dataset.get("annotations", [])
            self.unsaved_changes = True
            try:
                self.class_selector.populate_categories(self.coco_data["categories"])
            except Exception:
                pass
            self._display_image_with_annotations()
        except Exception as e:
            QMessageBox.critical(self, self.tr("error_title"), self.tr("error_remap_failed", e=e))
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _remove_current_image(self):
        # Check if there's a current image selected
        if self.current_image_id == -1 or not self.coco_data.get("images"):
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_current_image"))
            return

        # Remove the image from coco_data
        images = self.coco_data.get("images", [])
        self.coco_data["images"] = [img for img in images if img.get("id") != self.current_image_id]

        # Remove all annotations for this image
        annotations = self.coco_data.get("annotations", [])
        self.coco_data["annotations"] = [ann for ann in annotations if ann.get("image_id") != self.current_image_id]

        self.unsaved_changes = True

        # Clear the current image display
        self.image = None
        self.current_image_id = -1
        self.current_image_path = None
        self._display_image_with_annotations()

        # Clear image info pane
        self.image_info_pane.clear()

        # Update the image selector widget
        if hasattr(self, "image_selector_widget"):
            self.image_selector_widget.set_coco_data(self.coco_data, self.coco_json_path)

        self.statusBar().showMessage(self.tr("status_current_image_removed"), 3000)

    def _remove_unannotated_images(self):
        if not self.coco_json_path:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_dataset_loaded"))
            return

        self.coco_data = remove_unannotated_images_coco(self.coco_data)
        self.unsaved_changes = True

        # Clear current image if it was removed
        current_image_exists = any(img.get("id") == self.current_image_id for img in self.coco_data.get("images", []))
        if not current_image_exists:
            self.image = None
            self.current_image_id = -1
            self.current_image_path = None
            # Clear image info pane
            self.image_info_pane.clear()

        # Update the image selector widget
        if hasattr(self, "image_selector_widget"):
            self.image_selector_widget.set_coco_data(self.coco_data, self.coco_json_path)

        # Refresh the display
        self._display_image_with_annotations()

        self.statusBar().showMessage(self.tr("status_unannotated_images_removed"), 3000)

    def _convert_to_rle(self):
        self.coco_data = convert_coco_mask_type(self.coco_data, None, "rle")
        self.unsaved_changes = True
        self._display_image_with_annotations()
        self.statusBar().showMessage(self.tr("status_converted_masks_to_rle"), 3000)

    def _convert_to_polygons(self):
        self.coco_data = convert_coco_mask_type(self.coco_data, None, "polygons")
        self.unsaved_changes = True
        self._display_image_with_annotations()
        self.statusBar().showMessage(self.tr("status_converted_masks_to_polygons"), 3000)

    def _remove_segmentation_masks(self):
        self.coco_data = remove_segmentation_masks_from_coco(self.coco_data)
        self.unsaved_changes = True
        self._display_image_with_annotations()
        self.statusBar().showMessage(self.tr("status_segmentation_masks_removed"), 3000)

    def _add_bboxes_from_segmentation(self):

        dlg = AddBboxesFromSegmentationDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            options = dlg.get_options()
            image_ids = [self.current_image_id] if options["current_image_only"] and self.current_image_id != -1 else None

            self.coco_data = add_bboxes_from_segmentation(coco_input=self.coco_data, overwrite_existing_bboxes=options["overwrite_existing_bboxes"], image_ids=image_ids)
            self.unsaved_changes = True
            self._display_image_with_annotations()
            self.statusBar().showMessage(self.tr("status_bboxes_added_from_segmentations"), 3000)

    def _add_bboxes_from_segmentation_current_overwrite(self):
        if not self.coco_json_path:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_dataset_loaded"))
            return
        if self.current_image_id == -1:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_current_image"))
            return

        try:
            self.coco_data = add_bboxes_from_segmentation(
                coco_input=self.coco_data,
                overwrite_existing_bboxes=True,
                image_ids=[self.current_image_id],
            )
            self.unsaved_changes = True
            self._display_image_with_annotations()
            self.statusBar().showMessage(self.tr("status_bboxes_added_from_segmentations"), 3000)
        except Exception as e:
            QMessageBox.critical(self, self.tr("error_title"), str(e))

    def _change_categories_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("select_coco_json_file_dialog_title"), "", self.tr("json_files_filter"))
        if not file_path:
            return
        try:
            with open(file_path) as f:
                data = json.load(f)
            categories = data.get("categories", [])
            if not categories:
                QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_categories_found_in_file"))
                return
            self.coco_data["categories"] = categories
            self.coco_data["annotations"] = []
            self.unsaved_changes = True
            self.class_selector.populate_categories(self.coco_data["categories"])
            self.class_selector._unselect_current()
            self._display_image_with_annotations()
            self.statusBar().showMessage(self.tr("status_categories_imported_and_annotations_cleared"), 3000)
        except Exception as e:
            QMessageBox.critical(self, self.tr("error_title"), self.tr("error_failed_to_load_categories_from_file", e=e))

    def _create_toolbar(self):
        self.main_toolbar = QToolBar(self.tr("main_toolbar"), movable= False)
        self.addToolBar(self.main_toolbar)
        self.main_toolbar.addAction(self.zoom_in_action)
        self.main_toolbar.addAction(self.zoom_out_action)
        self.main_toolbar.addAction(self.fit_to_view_action)

        self.edit_toolbar = QToolBar(self.tr("edit_toolbar"), movable= False)
        self.addToolBar(self.edit_toolbar)
        self.edit_toolbar.addAction(self.view_mode_action)
        self.edit_toolbar.addAction(self.add_manual_mode_action)
        self.edit_toolbar.addAction(self.add_ai_mode_action)
        self.edit_toolbar.addAction(self.add_ai_multi_point_mode_action)
        self.edit_toolbar.addAction(self.remove_mode_action)
        self.edit_toolbar.addAction(self.edit_mode_action)
        self.edit_toolbar.addAction(self.change_mode_action)
        self.edit_toolbar.addAction(self.suggest_mode_action)
        self.edit_toolbar.setStyleSheet("""
            QToolButton#qt_toolbar_ext_button {
                background-color: #ffffff;
            }
            QToolButton#qt_toolbar_ext_button:hover {
                background-color: #34495e;
            }
        """)

        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.addAction(self.view_mode_action)
        self.mode_action_group.addAction(self.add_manual_mode_action)
        self.mode_action_group.addAction(self.add_ai_mode_action)
        self.mode_action_group.addAction(self.add_ai_multi_point_mode_action)
        self.mode_action_group.addAction(self.remove_mode_action)
        self.mode_action_group.addAction(self.edit_mode_action)
        self.mode_action_group.addAction(self.change_mode_action)
        self.mode_action_group.addAction(self.suggest_mode_action)
        self.mode_action_group.setExclusive(True)

    def _set_add_manual_mode(self):
        self.mode = "add-manual"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_add_manual_mode"), 2000)

    def _set_add_ai_mode(self):
        if self.segmenter.model_type is None:
            self.mode = DEFAULT_MODE
            self.add_ai_mode_action.setChecked(False)
            self.view_mode_action.setChecked(True)
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_segmentation_model_loaded"))

        self.mode = "add-AI"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_add_AI_mode"), 2000)

    def _set_add_ai_multi_point_mode(self):
        if self.segmenter.model_type is None:
            self.mode = DEFAULT_MODE
            self.add_ai_multi_point_mode_action.setChecked(False)
            self.view_mode_action.setChecked(True)
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_segmentation_model_loaded"))
            return

        self.mode = "add-AI-multi-point"
        self.add_ai_multi_point_controller.reset_points()
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_add_ai_multi_point_mode"), 2000)

    def _set_remove_mode(self):
        self.mode = "remove"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_remove_mode"), 2000)

    def _set_edit_mode(self):
        self.mode = "edit"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_edit_mode"), 2000)

    def _set_change_mode(self):
        self.mode = "change"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_change_mode"), 2000)

    def _set_view_mode(self):
        self.mode = "view"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_view_mode"), 2000)

    def _set_suggest_mode(self):
        self.mode = "suggest"
        self.unsetCursor()
        self.statusBar().showMessage(self.tr("status_suggest_mode"), 2000)

    def fit_to_view(self):
        if not self.image:
            return

        view_size = self.scroll_area.size()
        image_size = self.image.size

        width_ratio = view_size.width() / image_size[0]
        height_ratio = view_size.height() / image_size[1]

        self.zoom_scale = min(width_ratio, height_ratio)
        self._display_image_with_annotations()

    def _adjust_zoom(self, factor):
        self.zoom_scale *= factor
        self._display_image_with_annotations()

    def _save_annotations(self):
        # Removed check for presence of annotations to allow saving empty datasets

        # Just dump current self.coco_data (already up to date) to disk
        file_path = self.coco_json_path
        if not file_path:
            QMessageBox.critical(self, self.tr("error_saving_annotations_title"), self.tr("error_coco_path_missing"))
            return

        try:
            with open(file_path, "w") as f:
                json.dump(self.coco_data, f, indent=4)
            self.unsaved_changes = False
            self.statusBar().showMessage(self.tr("status_annotations_saved", file_path=file_path), 0)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("error_saving_annotations_title"),
                self.tr("error_saving_annotations_message", file_path=file_path, e=e),
            )

    def _save_annotations_as(self):
        # Prompt for new file path and then save using that path, updating self.coco_json_path
        # Removed check for presence of annotations to allow saving empty datasets
        new_path, _ = QFileDialog.getSaveFileName(self, self.tr("save_annotations_as_action"), self.coco_json_path or "", self.tr("json_files_filter"))
        if not new_path:
            return
        # Ensure .json extension
        if not new_path.lower().endswith(".json"):
            new_path += ".json"
        try:
            with open(new_path, "w") as f:
                json.dump(self.coco_data, f, indent=4)
            self.coco_json_path = new_path
            self.unsaved_changes = False
            # Update image selector with new path context
            self.image_selector_widget.set_coco_data(self.coco_data, self.coco_json_path)
            self.statusBar().showMessage(self.tr("status_annotations_saved", file_path=new_path), 0)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("error_saving_annotations_title"),
                self.tr("error_saving_annotations_message", file_path=new_path, e=e),
            )

    def _display_annotated_image(self, annotated_np: np.ndarray):
        """Display an annotated RGB numpy image on the QLabel with current zoom."""
        if annotated_np is None:
            return
        h, w, c = annotated_np.shape
        if c != 3:
            return
        qimage = QImage(annotated_np.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        transformation_mode = Qt.TransformationMode.FastTransformation if self.interpolation_mode == "nearest" else Qt.TransformationMode.SmoothTransformation
        scaled_pixmap = pixmap.scaled(
            int(w * self.zoom_scale),
            int(h * self.zoom_scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            transformation_mode,
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()

    def _display_image_with_annotations(self, overlay_masks=None, temp_bbox=None):
        """Wrapper kept for backward compatibility with existing call sites."""
        if not self.image:
            return
        anns = self._get_current_image_annotations()
        # If overlay_masks provided, create a temporary composited image
        image = self.image
        if overlay_masks is not None:
            image = overlay_candidate_segmentations(image, overlay_masks)

        if self.show_annotations_action.isChecked():
            annotated = draw_annotations_on_image(image, anns, self.coco_data.get("categories", []), draw_confidence=True, draw_label=self.draw_labels_action.isChecked(), segmentation_opacity=self.segmentation_opacity_selector_widget.get_opacity(), bbox_thickness=self.bbox_thickness_selector_widget.get_thickness(), category_colors=self.category_colors)
        else:
            annotated = np.array(image)
        annotated = self._overlay_current_suggestion(annotated)
        if self.mode == "add-AI-multi-point" and self.add_ai_multi_point_controller.has_points():
            positive_points, negative_points = self.add_ai_multi_point_controller.get_points()
            if positive_points:
                seg_w, seg_h = self.segmentation_inference_resolution_widget.get_resolution()
                masks = self.segmenter.segment_points(self.image, positive_points, negative_points, size=(seg_h, seg_w))
                if masks is not None and len(masks) > 0:
                    mask = masks[0].cpu().numpy().astype(np.uint8)
                    annotated = draw_segmentation_mask(annotated, mask, color_hex="#00FFFF", alpha=0.5)
            for px, py in positive_points:
                draw_plus(annotated, px, py, (0, 255, 0), 10)
            for px, py in negative_points:
                draw_minus(annotated, px, py, (255, 0, 0), 10)
        if temp_bbox:
            start_point, end_point = temp_bbox
            # Convert label coordinates to image coordinates
            x1 = int(min(start_point.x(), end_point.x()) / self.zoom_scale)
            y1 = int(min(start_point.y(), end_point.y()) / self.zoom_scale)
            x2 = int(max(start_point.x(), end_point.x()) / self.zoom_scale)
            y2 = int(max(start_point.y(), end_point.y()) / self.zoom_scale)
            width = x2 - x1
            height = y2 - y1
            if width > 0 and height > 0:
                draw_dashed_bbox_with_label(annotated, x1, y1, width, height, confidence=None, class_label="")
        self._display_annotated_image(annotated)

    def _overlay_current_suggestion(self, image_np: np.ndarray) -> np.ndarray:
        if image_np is None:
            return image_np  # type: ignore[return-value]
        # Draw current suggestion (dashed) if active and for this image
        if self.suggestion_mode_active and self.current_suggestion and self.current_suggestion.get("image_id") == self.current_image_id:
            bbox = self.current_suggestion.get("bbox")
            if bbox:
                x, y, w, h = bbox
                confidence = self.current_suggestion.get("bbox_confidence")
                cat_id = self.current_suggestion.get("category_id")
                categories_list = self.coco_data.get("categories", [])
                class_label = next((c.get("name") for c in categories_list if c.get("id") == cat_id), str(cat_id))
                if self.draw_labels_action.isChecked():
                    draw_dashed_bbox_with_label(image_np, x, y, w, h, confidence, class_label)
                else:
                    draw_dashed_bbox_with_label(image_np, x, y, w, h, None, "")
        return image_np

    def mousePressEvent(self, event):
        # Support navigating through images with mouse back/forward buttons.
        if event.button() == Qt.MouseButton.BackButton:
            if self.suggestion_mode_active:
                self._previous_suggestion()
            elif self.image_selector_widget:
                self.image_selector_widget.prev_image()
            event.accept()
            return
        if event.button() == Qt.MouseButton.ForwardButton:
            if self.suggestion_mode_active:
                self._advance_suggestion()
            elif self.image_selector_widget:
                self.image_selector_widget.next_image()
            event.accept()
            return

        if self.image and (event.button() == Qt.MouseButton.LeftButton or (self.mode == "add-AI-multi-point" and event.button() == Qt.MouseButton.RightButton)):
            if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
                label_point = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                if self.image_label.rect().contains(label_point):
                    if self.mode == "view":
                        pass  # Do nothing in view mode
                    elif self.mode == "add-manual":
                        self.add_manual_controller.start_drawing(label_point)
                    elif self.mode == "add-AI":
                        self.add_ai_controller.handle_press(label_point)
                    elif self.mode == "add-AI-multi-point":
                        self.add_ai_multi_point_controller.start_point(label_point, event.button())
                    elif self.mode == "remove":
                        self.remove_annotation_controller.handle_press(label_point)
                    elif self.mode == "edit":
                        self.edit_bbox_controller.handle_press(label_point)
                    elif self.mode == "change":
                        self.change_annotation_controller.handle_press(label_point)
                    elif self.mode == "suggest":
                        # Start resize if near edge
                        if self.current_suggestion and self.current_suggestion.get("image_id") == self.current_image_id:
                            handle = self._get_handle_for_bbox_point(self.current_suggestion["bbox"], label_point)
                            if handle:
                                self._suggestion_editing_handle = handle
                                self._suggestion_drag_start_pos = label_point
                                self._suggestion_original_bbox = self.current_suggestion["bbox"].copy()

    def mouseMoveEvent(self, event):
        if not (self.image and self.image_label.pixmap() and not self.image_label.pixmap().isNull()):
            super().mouseMoveEvent(event)
            return

        label_point = self.image_label.mapFromGlobal(event.globalPosition().toPoint())

        if self.mode == "view":
            self.unsetCursor()  # Ensure cursor is unset in view mode
            pass  # Do nothing in view mode
        elif self.mode == "edit":
            self.edit_bbox_controller.handle_move(label_point)
        elif self.mode == "suggest" and self.current_suggestion:
            # Update suggestion bounding box position while dragging
            if self._suggestion_editing_handle and self._suggestion_drag_start_pos and self._suggestion_original_bbox:
                if not self.image_label.rect().contains(label_point):
                    super().mouseMoveEvent(event)
                    return

                delta_x = (label_point.x() - self._suggestion_drag_start_pos.x()) / self.zoom_scale
                delta_y = (label_point.y() - self._suggestion_drag_start_pos.y()) / self.zoom_scale

                bbox = self._suggestion_original_bbox.copy()
                min_size = 5  # Minimum size in image pixels

                if self._suggestion_editing_handle == "left":
                    new_x = round(self._suggestion_original_bbox[0] + delta_x)
                    new_w = round(self._suggestion_original_bbox[2] - delta_x)
                    if new_w >= min_size:
                        bbox[0] = new_x
                        bbox[2] = new_w
                elif self._suggestion_editing_handle == "right":
                    new_w = round(self._suggestion_original_bbox[2] + delta_x)
                    if new_w >= min_size:
                        bbox[2] = new_w
                elif self._suggestion_editing_handle == "top":
                    new_y = round(self._suggestion_original_bbox[1] + delta_y)
                    new_h = round(self._suggestion_original_bbox[3] - delta_y)
                    if new_h >= min_size:
                        bbox[1] = new_y
                        bbox[3] = new_h
                elif self._suggestion_editing_handle == "bottom":
                    new_h = round(self._suggestion_original_bbox[3] + delta_y)
                    if new_h >= min_size:
                        bbox[3] = new_h

                self.current_suggestion["bbox"] = bbox
                self._display_image_with_annotations()
            else:  # Hovering over suggestion bbox
                handle = self._get_handle_for_bbox_point(self.current_suggestion["bbox"], label_point)
                if handle in ["top", "bottom"]:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                elif handle in ["left", "right"]:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.unsetCursor()
        elif self.mode == "add-manual" and self.add_manual_controller.drawing_bbox_start_pos:
            if not self.image_label.rect().contains(label_point):
                super().mouseMoveEvent(event)
                return
            self.add_manual_controller.update_drawing(label_point)
        else:
            self.unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "view":
            pass  # Do nothing in view mode
        elif event.button() == Qt.MouseButton.LeftButton and self.edit_bbox_controller.editing_annotation:
            self.edit_bbox_controller.handle_release()
        # In suggestion mode just finalize resize; do not auto-accept
        elif event.button() == Qt.MouseButton.LeftButton and self.mode == "suggest":
            if self._suggestion_editing_handle:
                self._suggestion_editing_handle = None
                self._suggestion_drag_start_pos = None
                self._suggestion_original_bbox = None
                self.unsetCursor()
        elif event.button() == Qt.MouseButton.LeftButton and self.mode == "add-manual":
            end_point = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            self.add_manual_controller.finish_drawing(end_point)

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and (event.key() == Qt.Key.Key_Equal or event.key() == Qt.Key.Key_Plus):
            self._zoom_at_point(1.2, self.scroll_area.viewport().mapFromGlobal(QCursor.pos()))
            event.accept()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Minus:
            self._zoom_at_point(1 / 1.2, self.scroll_area.viewport().mapFromGlobal(QCursor.pos()))
            event.accept()
        elif self.mode == "add-AI-multi-point":
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.add_ai_multi_point_controller.finish_and_add()
            elif event.key() == Qt.Key_Escape:
                self.add_ai_multi_point_controller.cancel()
            event.accept()
        elif self.mode == "add-manual":
            # If the user presses ESC while drawing a manual bbox, cancel it.
            if event.key() == Qt.Key_Escape:
                # Forward to the controller which knows how to cancel drawing
                try:
                    self.add_manual_controller.cancel()
                except Exception:
                    # Be conservative: don't let a cancellation error block the app
                    pass
                event.accept()
        else:
            super().keyPressEvent(event)

    def _get_current_image_annotations(self):
        """Return annotation dict references for the current image id."""
        if self.current_image_id == -1:
            return []
        return [ann for ann in self.coco_data.get("annotations", []) if ann.get("image_id") == self.current_image_id]

    def _zoom_at_point(self, factor, widget_point):
        if not self.image:
            return

        # Point on the image_label (potentially very large)
        old_scale_label_point_x = widget_point.x() + self.scroll_area.horizontalScrollBar().value()
        old_scale_label_point_y = widget_point.y() + self.scroll_area.verticalScrollBar().value()

        # Point on the original image (unscaled)
        image_point_x = old_scale_label_point_x / self.zoom_scale
        image_point_y = old_scale_label_point_y / self.zoom_scale

        # Apply zoom
        self.zoom_scale *= factor
        self._display_image_with_annotations()  # This resizes the label

        # New coordinates of the image point on the scaled label
        new_scale_label_point_x = image_point_x * self.zoom_scale
        new_scale_label_point_y = image_point_y * self.zoom_scale

        # Scroll to keep the image point under the mouse pointer
        self.scroll_area.horizontalScrollBar().setValue(int(new_scale_label_point_x - widget_point.x()))
        self.scroll_area.verticalScrollBar().setValue(int(new_scale_label_point_y - widget_point.y()))

    def _get_max_annotation_id(self):
        """Return the current maximum annotation id from self.coco_data."""
        max_annotation_id = 0
        if self.coco_data and "annotations" in self.coco_data:
            for ann in self.coco_data["annotations"]:
                if "id" in ann and isinstance(ann["id"], int):
                    max_annotation_id = max(max_annotation_id, ann["id"])
        return max_annotation_id

    def _segment_existing_bboxes(self):
        # Minimal integration without new translation keys
        if not self.coco_data or not self.coco_json_path:
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_dataset_loaded"))
            return
        # Persist current in-memory annotations to file first (so segmentation works on latest state)
        try:
            with open(self.coco_json_path, "w") as f:
                json.dump(self.coco_data, f)
        except Exception as e:
            QMessageBox.critical(self, self.tr("error_title"), self.tr("error_failed_to_write_dataset_before_segmentation", e=e))
            return
        dialog = SegmentBBoxesDialog(self, self.tr)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.statusBar().showMessage(self.tr("status_generating_segmentations"), 0)
            updated_dataset = dialog.segment_dataset(self.coco_data, self.coco_json_path, self.device_selector.get_selected_device())
            if updated_dataset:
                self.coco_data = updated_dataset
                self.unsaved_changes = True
                # Refresh UI components referencing dataset
                # self.class_selector.populate_categories(self.coco_data.get("categories", []))
                # self.image_selector_widget.set_coco_data(self.coco_data, self.coco_json_path)
                assert self.current_image_id != -1
                self._display_image_with_annotations()
                self.statusBar().showMessage(self.tr("status_segmentation_complete_in_memory"), 4000)
            else:
                self.statusBar().showMessage(self.tr("status_segmentation_produced_no_changes"), 4000)
        except Exception as e:
            QMessageBox.critical(self, self.tr("error_title"), self.tr("status_segmentation_failed", e=e))
            self.statusBar().showMessage(self.tr("status_segmentation_failed"), 4000)

    # Event filter to capture Ctrl + Wheel on the scroll area viewport for zooming
    def eventFilter(self, obj, event):
        if obj == self.scroll_area.viewport() and event.type() == QEvent.Type.Wheel:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Determine zoom direction: positive angleDelta().y() means wheel up
                delta = event.angleDelta().y()
                if delta != 0 and self.image:
                    factor = 1.2 if delta > 0 else 1 / 1.2
                    # Map current cursor position within the viewport for focal zoom
                    cursor_point = self.scroll_area.viewport().mapFromGlobal(QCursor.pos())
                    self._zoom_at_point(factor, cursor_point)
                    event.accept()
                    return True  # Consume event (prevent scroll)
        return super().eventFilter(obj, event)

    def _start_suggestion_mode(self):
        if not self.coco_json_path or not os.path.isfile(self.coco_json_path):
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("warning_no_dataset_loaded"))
            return
        if self.suggestion_mode_active:
            return
        dialog = SuggestDetectionsDialog(self, categories=self.coco_data.get("categories", []))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.statusBar().showMessage(self.tr("status_generating_suggestions"), 0)
        suggestions = dialog.generate_suggestions(self.coco_data, self.coco_json_path, self.device_selector.get_selected_device())
        if not suggestions:
            self.statusBar().showMessage(self.tr("status_no_suggestions_found"), 3000)
            return
        self.suggestions = suggestions
        self.current_suggestion_index = -1
        self.current_suggestion = None
        self.suggestion_mode_active = True
        self.suggest_mode_action.setEnabled(True)
        self._set_suggest_mode()
        self._swap_in_suggestion_control_widget()
        self._advance_suggestion()
        self.statusBar().showMessage(self.tr("status_suggestion_mode_started"), 3000)

    def _swap_in_suggestion_control_widget(self):
        if self.suggestion_control_widget is None:
            self.suggestion_control_widget = SuggestionControlWidget(tr_func=self.tr, accept_callback=self._accept_current_suggestion, previous_callback=self._previous_suggestion, reject_callback=self._reject_current_suggestion, finish_callback=self._end_suggestion_mode, parent=self)
        if self.image_selector_widget is not None:
            self.image_selector_widget.setParent(None)
        self._right_pane_layout.addWidget(self.suggestion_control_widget)

    def _restore_image_selector_widget(self):
        if self.suggestion_control_widget is not None:
            self.suggestion_control_widget.setParent(None)
        if self.image_selector_widget is not None:
            self._right_pane_layout.addWidget(self.image_selector_widget)

    def _advance_suggestion(self):
        self.current_suggestion_index += 1
        if self.current_suggestion_index >= len(self.suggestions):
            self._end_suggestion_mode()
            return
        self._update_current_suggestion()

    def _previous_suggestion(self):
        if self.current_suggestion_index <= 0:
            self.statusBar().showMessage(self.tr("already_at_first_suggestion"), 2000)
            return
        self.current_suggestion_index -= 1
        self._update_current_suggestion()

    def _update_current_suggestion(self):
        raw = self.suggestions[self.current_suggestion_index]
        self.current_suggestion = dict(raw)
        image_id = self.current_suggestion.get("image_id")
        if image_id is not None:
            self._load_image_by_id(image_id)
        if self.suggestion_control_widget:
            self.suggestion_control_widget.update_progress(self.current_suggestion_index + 1, len(self.suggestions))

        bbox = self.current_suggestion.get("bbox")
        if bbox and self.zoom_on_suggestion_level > 1:
            self._zoom_to_bbox(bbox, max_zoom=self.zoom_on_suggestion_level)
        else:
            self.fit_to_view()

    def _zoom_to_bbox(self, bbox, max_zoom=2.0):
        if not self.image:
            return

        view_size = self.scroll_area.viewport().size()
        bbox_w, bbox_h = bbox[2], bbox[3]

        if bbox_w == 0 or bbox_h == 0:
            self.fit_to_view()
            return

        width_ratio = view_size.width() / bbox_w
        height_ratio = view_size.height() / bbox_h

        # Add some padding
        scale = min(width_ratio, height_ratio) * 0.9

        # Cap zoom at max_zoom
        self.zoom_scale = min(scale, max_zoom)

        self._display_image_with_annotations()

        # Center view on bbox
        bbox_center_x = bbox[0] + bbox_w / 2
        bbox_center_y = bbox[1] + bbox_h / 2

        scaled_center_x = bbox_center_x * self.zoom_scale
        scaled_center_y = bbox_center_y * self.zoom_scale

        viewport_center_x = view_size.width() / 2
        viewport_center_y = view_size.height() / 2

        self.scroll_area.horizontalScrollBar().setValue(int(scaled_center_x - viewport_center_x))
        self.scroll_area.verticalScrollBar().setValue(int(scaled_center_y - viewport_center_y))

    def _accept_current_suggestion(self):
        if not self.current_suggestion:
            return
        selection = self.class_selector.get_current_selection(supress_warning=True)
        if selection and selection[0] is not None:
            _, self.current_suggestion["category_id"] = selection
        del self.current_suggestion["bbox_confidence"]

        # Type ignore: runtime stores dict; static model may expect specific annotation dataclass
        self.coco_data.setdefault("annotations", []).append(self.current_suggestion)  # type: ignore[arg-type]
        self.unsaved_changes = True
        self.statusBar().showMessage(self.tr("status_suggestion_accepted"), 1200)
        self._advance_suggestion()

    def _reject_current_suggestion(self):
        if not self.current_suggestion:
            return
        self._advance_suggestion()

    def _end_suggestion_mode(self):
        if not self.suggestion_mode_active:
            return
        self.suggest_mode_action.setEnabled(False)
        self.suggestion_mode_active = False
        self.mode = DEFAULT_MODE
        self.current_suggestion = None
        self.suggestions = []
        self.current_suggestion_index = -1
        self._restore_image_selector_widget()
        self._display_image_with_annotations()
        self.statusBar().showMessage(self.tr("status_suggestion_mode_finished"), 3000)

    def _load_image_by_id(self, image_id: int):
        meta = next((img for img in self.coco_data.get("images", []) if img.get("id") == image_id), None)
        if not meta:
            return
        file_name = meta.get("file_name")
        if not file_name:
            return
        root = os.path.dirname(self.coco_json_path) if self.coco_json_path else ""
        path = os.path.join(root, file_name)
        try:
            self.image = Image.open(path).convert("RGB")
            annotations = [ann for ann in self.coco_data.get("annotations", []) if ann.get("image_id") == image_id]
            self._on_image_selected(path, self.image, annotations, image_id)
        except Exception:
            pass

    def _get_handle_for_bbox_point(self, bbox, point):
        x, y, w, h = bbox
        scaled_x = x * self.zoom_scale
        scaled_y = y * self.zoom_scale
        scaled_w = w * self.zoom_scale
        scaled_h = h * self.zoom_scale
        handle_margin = 2 * self.zoom_scale
        on_left_edge = abs(point.x() - scaled_x) < handle_margin and scaled_y <= point.y() <= scaled_y + scaled_h
        on_right_edge = abs(point.x() - (scaled_x + scaled_w)) < handle_margin and scaled_y <= point.y() <= scaled_y + scaled_h
        on_top_edge = abs(point.y() - scaled_y) < handle_margin and scaled_x <= point.x() <= scaled_x + scaled_w
        on_bottom_edge = abs(point.y() - (scaled_y + scaled_h)) < handle_margin and scaled_x <= point.x() <= scaled_x + scaled_w
        if on_left_edge:
            return "left"
        if on_right_edge:
            return "right"
        if on_top_edge:
            return "top"
        if on_bottom_edge:
            return "bottom"
        return None

    def _on_segmentation_model_switched(self, action):
        new_model_key = action.data()
        self.segmenter = Segmenter(model_type=new_model_key, device=self.device_selector.get_selected_device())

        if self.statusBar():
            self.statusBar().showMessage(self.tr("status_model_loaded", model_name=new_model_key), 2000)

    def _on_device_switched(self, new_device):
        self.segmenter = Segmenter(model_type=self.segmentation_model_selector.get_selected_model(), device=new_device)
        self.statusBar().showMessage(self.tr("status_device_switched", device_name=new_device), 2000)

    def _create_unlabeled_coco_dataset(self):
        dlg = CreateUnlabeledCocoDialog(self)
        dlg.exec()

    def _open_segmentation_opacity_dialog(self):
        dlg = self.segmentation_opacity_selector_widget
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._display_image_with_annotations()

    def _open_bbox_thickness_dialog(self):
        dlg = self.bbox_thickness_selector_widget
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._display_image_with_annotations()

    def _show_dataset_info(self):
        if not self.coco_json_path:
            QMessageBox.information(self, self.tr("info_title"), self.tr("warning_no_dataset_loaded"))
            return
        dlg = DatasetInfoDialog(self, coco_data=self.coco_data, tr_func=self.tr)
        dlg.exec()

    def _split_dataset(self):
        if not self.coco_json_path:
            QMessageBox.warning(self, self.tr("warning"), self.tr("no_dataset_loaded"))
            return
        dialog = SplitCocoDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_params()
            if not params["output_dir"]:
                QMessageBox.warning(self, self.tr("warning"), self.tr("output_dir_required"))
                return
            try:
                image_dir = os.path.dirname(self.coco_json_path)
                original_size, train_size, valid_size, test_size = split_and_save_coco(image_dir=image_dir, annotations_path=self.coco_json_path, output_dir=params["output_dir"], train_split=params["train_split"], validation_split=params["validation_split"], random_state=params["random_state"], shuffle=params["shuffle"])
                QMessageBox.information(self, self.tr("success"), self.tr("dataset_split_success").format(original=original_size, train=train_size, valid=valid_size, test=test_size))
            except Exception as e:
                QMessageBox.critical(self, self.tr("error"), str(e))

    def _merge_datasets(self):
        from panopticon.widgets.merge_coco_dialog import MergeCocoDialog

        dialog = MergeCocoDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.merged_dataset_path:
            self.load_coco_dataset_from_path(dialog.merged_dataset_path)

    def closeEvent(self, event):
        if self.unsaved_changes:
            reply = QMessageBox.question(self, self.tr("unsaved_changes_title"), self.tr("unsaved_changes_message"), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes:
                self._save_annotations()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("../images/pan_annotator_logo_transparent.png"))

    annotator = PanAnnotator()
    annotator.show()
    sys.exit(app.exec())
