import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from panopticon.data_engine.segment_coco import generate_missing_segmentations
from panopticon.widgets.model_selector import ModelSelector


class SegmentBBoxesDialog(QDialog):
    """Dialog for generating segmentation masks for existing bounding boxes.

    No new translation keys added intentionally.
    """

    def __init__(self, parent, tr_func):
        assert parent is not None
        super().__init__(parent)
        self.tr = tr_func
        self.setWindowTitle(self.tr("segment_existing_bboxes_dialog_title"))
        self.main_layout = QVBoxLayout(self)

        description_label = QLabel(self.tr("segment_existing_bboxes_dialog_description"))
        description_label.setWordWrap(True)
        self.main_layout.addWidget(description_label)

        # Model Type (reuse ModelSelector)
        self.main_layout.addWidget(QLabel(self.tr("model_type_label")))
        parent_widget = parent
        status_bar = getattr(parent_widget, "statusBar", lambda: None)()
        self.model_selector = ModelSelector(parent_widget, self.tr, status_bar, auto_load_save=False, mode="segmentation")

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
        self.main_layout.addWidget(self.model_type_combo)

        # Batch Size
        self.main_layout.addWidget(QLabel(self.tr("batch_size_label")))
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 256)
        self.batch_size_input.setValue(1)
        self.main_layout.addWidget(self.batch_size_input)

        # Inference Size
        self.main_layout.addWidget(QLabel(self.tr("inference_size_label")))
        self.size_input = QSpinBox()
        self.size_input.setRange(64, 4096)
        self.size_input.setValue(1024)
        self.main_layout.addWidget(self.size_input)

        # Overwrite existing segmentations
        self.overwrite_checkbox = QCheckBox(self.tr("overwrite_existing_segmentations_checkbox"))
        self.overwrite_checkbox.setChecked(False)
        self.main_layout.addWidget(self.overwrite_checkbox)

        # Mask Type
        self.main_layout.addWidget(QLabel(self.tr("mask_type_label")))
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItem("RLE", userData="RLE")
        self.mask_type_combo.addItem("Polygon", userData="polygon")
        self.main_layout.addWidget(self.mask_type_combo)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("cancel_button"))
        self.main_layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(self.main_layout)

    def get_arguments(self):
        return {
            "model_type": self.model_selector.get_selected_model(),
            "batch_size": self.batch_size_input.value(),
            "size": self.size_input.value(),
            "overwrite_existing": self.overwrite_checkbox.isChecked(),
            "mask_type": self.mask_type_combo.currentData(),
        }

    def segment_dataset(self, coco_data, coco_json_path, device):
        args = self.get_arguments()
        updated_dataset = generate_missing_segmentations(
            coco_input=coco_data,
            images_root=os.path.dirname(coco_json_path),
            output_json_path=None,  # in-memory only; we already have a file, but we'll reload to memory
            model_type=args["model_type"],
            batch_size=args["batch_size"],
            size=args["size"],
            device=device,
            overwrite_existing=args["overwrite_existing"],
            mask_type=args["mask_type"],
        )
        return updated_dataset
