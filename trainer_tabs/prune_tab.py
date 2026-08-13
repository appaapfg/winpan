import os

import torch
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from panopticon.util.prune_model import prune_conv2d_layers


class PruneTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()

        # Model path
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse")
        self.model_button.clicked.connect(self.browse_model)
        hbox = QHBoxLayout()
        hbox.addWidget(self.model_edit)
        hbox.addWidget(self.model_button)
        layout.addRow("Model Path:", hbox)

        # Amount
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.0, 1.0)
        self.amount_spin.setSingleStep(0.1)
        self.amount_spin.setValue(0.2)
        layout.addRow("Prune Amount:", self.amount_spin)

        # Method
        self.method_combo = QComboBox()
        self.method_combo.addItem("l1_unstructured")
        layout.addRow("Method:", self.method_combo)

        # Output path
        self.output_edit = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.output_edit)
        hbox2.addWidget(self.output_button)
        layout.addRow("Output Path:", hbox2)

        # Prune button
        self.prune_button = QPushButton("Start Pruning")
        self.prune_button.clicked.connect(self.start_pruning)
        layout.addRow(self.prune_button)

        self.setLayout(layout)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Model File", "", "Model Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model_edit.setText(file_path)

    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Pruned Model", "", "Model Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.output_edit.setText(file_path)

    def start_pruning(self):
        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please select a model file.")
            return
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return

        amount = self.amount_spin.value()
        method = self.method_combo.currentText()
        output_path = self.output_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Error", "Please specify an output path.")
            return

        try:
            model = torch.load(model_path)
            pruned_model = prune_conv2d_layers(model, amount=amount, method=method)
            torch.save(pruned_model, output_path)
            QMessageBox.information(self, "Success", f"Pruned model saved to {output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pruning failed: {str(e)}")
