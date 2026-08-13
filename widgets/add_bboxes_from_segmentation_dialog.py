from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)


class AddBboxesFromSegmentationDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(self.parent().tr("add_bboxes_from_segmentation_dialog_title"))
        self.resize(400, 200)

        layout = QVBoxLayout(self)

        description = QLabel("Add bounding boxes to annotations based on their segmentation masks.")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.overwrite_checkbox = QCheckBox(self.parent().tr("overwrite_existing_bboxes_checkbox"))
        self.overwrite_checkbox.setChecked(True)
        layout.addWidget(self.overwrite_checkbox)

        self.current_image_only_checkbox = QCheckBox(self.parent().tr("current_image_only_checkbox"))
        layout.addWidget(self.current_image_only_checkbox)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Hide initially, show during processing if needed
        layout.addWidget(self.progress_bar)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_options(self):
        return {"overwrite_existing_bboxes": self.overwrite_checkbox.isChecked(), "current_image_only": self.current_image_only_checkbox.isChecked()}
