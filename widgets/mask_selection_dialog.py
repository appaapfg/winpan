from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

# Local constant duplicated from pan_annotator ordering; keep in sync if changed there.
SEGMENTATION_COLOR_NAMES = ["GREEN", "YELLOW", "RED"]


class MaskSelectionDialog(QDialog):
    """Dialog allowing a user to select one of the generated segmentation masks.

    Notes:
        - The translation function comes from the parent window via parent().tr(...)
        - We duplicate SEGMENTATION_COLOR_NAMES locally to avoid circular imports.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use parent's translator if available
        self.setWindowTitle(self.parent().tr("select_mask_dialog_title") if parent else "Select Mask")
        self.v_layout = QVBoxLayout(self)
        self.setLayout(self.v_layout)

        self.mask_selector_label = QLabel(self.parent().tr("select_mask_label") if parent else "Select mask:")
        self.v_layout.addWidget(self.mask_selector_label)
        self.mask_combo_box = QComboBox()
        if parent and hasattr(parent, "tr"):
            self.mask_combo_box.addItems([self.parent().tr(f"color_{name.lower()}") for name in SEGMENTATION_COLOR_NAMES])
        else:
            self.mask_combo_box.addItems(SEGMENTATION_COLOR_NAMES)
        self.v_layout.addWidget(self.mask_combo_box)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        if parent and hasattr(parent, "tr"):
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.parent().tr("ok_button"))
            self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.parent().tr("cancel_button"))
        self.v_layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.chosen_mask_index = 0

    def accept(self):  # noqa: D401 (simple override)
        self.chosen_mask_index = self.mask_combo_box.currentIndex()
        super().accept()

    @staticmethod
    def get_mask_index(parent=None):
        dialog = MaskSelectionDialog(parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.chosen_mask_index, True
        return None, False
