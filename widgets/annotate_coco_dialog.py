from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
)

from panopticon.widgets.model_selector import ModelSelector


class AnnotateCocoDialog(QDialog):
    """Dialog for batch annotating a COCO dataset.

    Extracted from pan_annotator.py to its own module.
    """

    def __init__(self, parent, categories=None):
        assert parent is not None
        super().__init__(parent)
        self.setWindowTitle(self.parent().tr("annotate_coco_dialog_title"))
        self.main_layout = QVBoxLayout(self)

        # Model Type (using ModelSelector instead of plain line edit)
        self.main_layout.addWidget(QLabel(self.parent().tr("model_type_label")))
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

        # Sync combo changes back to ModelSelector actions
        def _on_combo_changed(index):
            selected_key = self.model_type_combo.itemData(index)
            for act in self.model_selector.get_actions():
                if act.data() == selected_key:
                    act.trigger()
                    break

        self.model_type_combo.currentIndexChanged.connect(_on_combo_changed)
        self.main_layout.addWidget(self.model_type_combo)

        # Threshold
        self.main_layout.addWidget(QLabel(self.parent().tr("threshold_label")))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(0.2)  # Default value
        self.main_layout.addWidget(self.threshold_input)

        # Batch Size
        self.main_layout.addWidget(QLabel(self.parent().tr("batch_size_label")))
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 256)
        self.batch_size_input.setValue(32)
        self.main_layout.addWidget(self.batch_size_input)

        # Width
        self.main_layout.addWidget(QLabel(self.parent().tr("width_label")))
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 8192)
        self.width_input.setValue(640)
        self.main_layout.addWidget(self.width_input)

        # Height
        self.main_layout.addWidget(QLabel(self.parent().tr("height_label")))
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 8192)
        self.height_input.setValue(640)
        self.main_layout.addWidget(self.height_input)

        # Object Categories
        self.main_layout.addWidget(QLabel(self.parent().tr("object_categories_label")))
        self.category_list_widget = QListWidget()
        self.category_list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        if categories:
            for cat_info in categories:
                item = QListWidgetItem(cat_info["name"])
                item.setData(Qt.ItemDataRole.UserRole, cat_info)
                item.setCheckState(Qt.CheckState.Checked)
                self.category_list_widget.addItem(item)
        self.main_layout.addWidget(self.category_list_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.parent().tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.parent().tr("cancel_button"))
        self.main_layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.setLayout(self.main_layout)

    def get_arguments(self):
        selected_categories = []
        for i in range(self.category_list_widget.count()):
            item = self.category_list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_categories.append(item.data(Qt.ItemDataRole.UserRole))

        return {
            "model_type": self.model_selector.get_selected_model(),
            "threshold": self.threshold_input.value(),
            "batch_size": self.batch_size_input.value(),
            "width": self.width_input.value(),
            "height": self.height_input.value(),
            "object_categories": selected_categories,
        }
