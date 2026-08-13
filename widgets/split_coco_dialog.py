from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class SplitCocoDialog(QDialog):
    """Dialog for splitting a COCO dataset into train, validation, and test sets."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(self.parent().tr("split_coco_dialog_title"))
        self.resize(600, 400)
        self.main_layout = QVBoxLayout(self)

        # Add a detailed description label at the top
        description_label = QLabel(self.parent().tr("split_coco_dialog_description"))
        description_label.setWordWrap(True)
        self.main_layout.addWidget(description_label)

        layout = QVBoxLayout()

        # Output directory
        layout.addWidget(QLabel(self.parent().tr("output_dir_label")))
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        self.browse_button = QPushButton(self.parent().tr("browse_button"))
        self.browse_button.clicked.connect(self._browse_output_dir)
        output_dir_layout.addWidget(self.browse_button)
        layout.addLayout(output_dir_layout)

        # Train split
        layout.addWidget(QLabel(self.parent().tr("train_split_label")))
        self.train_split_input = QDoubleSpinBox()
        self.train_split_input.setRange(0.0, 1.0)
        self.train_split_input.setSingleStep(0.1)
        self.train_split_input.setValue(0.7)
        layout.addWidget(self.train_split_input)

        # Validation split
        layout.addWidget(QLabel(self.parent().tr("validation_split_label")))
        self.validation_split_input = QDoubleSpinBox()
        self.validation_split_input.setRange(0.0, 1.0)
        self.validation_split_input.setSingleStep(0.1)
        self.validation_split_input.setValue(0.5)
        layout.addWidget(self.validation_split_input)

        # Random state
        layout.addWidget(QLabel(self.parent().tr("random_state_label")))
        self.random_state_input = QSpinBox()
        self.random_state_input.setRange(0, 1000000)
        self.random_state_input.setValue(42)
        layout.addWidget(self.random_state_input)

        # Shuffle
        self.shuffle_checkbox = QCheckBox(self.parent().tr("shuffle_label"))
        self.shuffle_checkbox.setChecked(True)
        layout.addWidget(self.shuffle_checkbox)

        self.main_layout.addLayout(layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.parent().tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.parent().tr("cancel_button"))
        self.main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, self.parent().tr("select_output_directory"))
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_params(self):
        return {"output_dir": self.output_dir_edit.text(), "train_split": self.train_split_input.value(), "validation_split": self.validation_split_input.value(), "random_state": self.random_state_input.value(), "shuffle": self.shuffle_checkbox.isChecked()}
