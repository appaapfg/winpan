import sys

import yaml
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


class ExportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Reference to the main window for menu bar, etc.

        # Tabs
        self.tabs = QTabWidget()

        # Basic tab
        self.basic_tab = QWidget()
        self.tabs.addTab(self.basic_tab, "Basic")
        self.setup_basic_tab()

        # Advanced tab
        self.advanced_tab = QWidget()
        self.tabs.addTab(self.advanced_tab, "Advanced")
        self.setup_advanced_tab()

        # Export export command button
        self.export_button = QPushButton("Export Export Command (for Developers)")
        self.export_button.clicked.connect(self.export_export_command)

        # Export button
        self.export_button_run = QPushButton("Start Export")
        self.export_button_run.clicked.connect(self.start_export)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.export_button)
        layout.addWidget(self.export_button_run)
        self.setLayout(layout)

    def load_export_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Load Config", "", "PanExporter Config Files (*.pan_exporter.yaml);;All Files (*)")
        if file_path:
            if not file_path.endswith(".pan_exporter.yaml"):
                QMessageBox.warning(self, "Invalid File", "Config file must end with '.pan_exporter.yaml'")
                return
            try:
                with open(file_path) as f:
                    config = yaml.safe_load(f)
                self.apply_config(config)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")

    def save_export_config(self):
        file_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Config", "", "PanExporter Config Files (*.pan_exporter.yaml);;All Files (*)")
        if file_path:
            if not file_path.endswith(".pan_exporter.yaml"):
                file_path += ".pan_exporter.yaml"
            try:
                config = self.collect_non_default()
                with open(file_path, "w") as f:
                    yaml.dump(config, f)
                QMessageBox.information(self, "Success", "Config saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {str(e)}")

    def collect_non_default(self):
        config = {}
        # Basic - always include if not default
        if self.model_edit.text() != "yolo11n.pt":
            config["model"] = self.model_edit.text()
        if self.format_combo.currentText() != "torchscript":
            config["format"] = self.format_combo.currentText()
        if self.imgsz_spin.value() != 640:
            config["imgsz"] = self.imgsz_spin.value()
        if self.device_edit.text() != "0":
            config["device"] = self.device_edit.text()
        if self.batch_spin.value() != 1:
            config["batch"] = self.batch_spin.value()

        # Advanced
        if not self.keras_use.isChecked():
            config["keras"] = self.keras_check.isChecked()
        if not self.optimize_use.isChecked():
            config["optimize"] = self.optimize_check.isChecked()
        if not self.half_use.isChecked():
            config["half"] = self.half_check.isChecked()
        if not self.int8_use.isChecked():
            config["int8"] = self.int8_check.isChecked()
        if not self.dynamic_use.isChecked():
            config["dynamic"] = self.dynamic_check.isChecked()
        if not self.simplify_use.isChecked():
            config["simplify"] = self.simplify_check.isChecked()
        if not self.opset_use.isChecked() and self.opset_spin.value() != -1:
            config["opset"] = self.opset_spin.value()
        if not self.workspace_use.isChecked() and self.workspace_spin.value() != -1.0:
            config["workspace"] = self.workspace_spin.value()
        if not self.nms_use.isChecked():
            config["nms"] = self.nms_check.isChecked()
        if not self.data_use.isChecked() and self.data_edit.text() != "coco8.yaml":
            config["data"] = self.data_edit.text()
        if not self.fraction_use.isChecked() and self.fraction_spin.value() != 1.0:
            config["fraction"] = self.fraction_spin.value()

        return config

    def apply_config(self, config):
        # Basic
        if "model" in config:
            self.model_edit.setText(config["model"])
        if "format" in config:
            index = self.format_combo.findText(config["format"])
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
        if "imgsz" in config:
            self.imgsz_spin.setValue(config["imgsz"])
        if "device" in config:
            self.device_edit.setText(str(config["device"]))
        if "batch" in config:
            self.batch_spin.setValue(config["batch"])

        # Advanced
        if "keras" in config:
            self.keras_check.setChecked(config["keras"])
            self.keras_use.setChecked(False)
        if "optimize" in config:
            self.optimize_check.setChecked(config["optimize"])
            self.optimize_use.setChecked(False)
        if "half" in config:
            self.half_check.setChecked(config["half"])
            self.half_use.setChecked(False)
        if "int8" in config:
            self.int8_check.setChecked(config["int8"])
            self.int8_use.setChecked(False)
        if "dynamic" in config:
            self.dynamic_check.setChecked(config["dynamic"])
            self.dynamic_use.setChecked(False)
        if "simplify" in config:
            self.simplify_check.setChecked(config["simplify"])
            self.simplify_use.setChecked(False)
        if "opset" in config:
            self.opset_spin.setValue(config["opset"])
            self.opset_use.setChecked(False)
        if "workspace" in config:
            self.workspace_spin.setValue(config["workspace"])
            self.workspace_use.setChecked(False)
        if "nms" in config:
            self.nms_check.setChecked(config["nms"])
            self.nms_use.setChecked(False)
        if "data" in config:
            self.data_edit.setText(config["data"])
            self.data_use.setChecked(False)
        if "fraction" in config:
            self.fraction_spin.setValue(config["fraction"])
            self.fraction_use.setChecked(False)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Model File", "", "Model Files (*.pt *.yaml);;All Files (*)")
        if file_path:
            self.model_edit.setText(file_path)

    def browse_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Data File", "", "YAML Files (*.yaml);;All Files (*)")
        if file_path:
            self.data_edit.setText(file_path)

    def setup_basic_tab(self):
        layout = QFormLayout()

        # model
        self.model_edit = QLineEdit("yolo11n.pt")
        self.model_button = QPushButton("Browse")
        self.model_button.clicked.connect(self.browse_model)
        hbox = QHBoxLayout()
        hbox.addWidget(self.model_edit)
        hbox.addWidget(self.model_button)
        layout.addRow("Model:", hbox)

        # format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["torchscript", "onnx", "openvino", "engine", "coreml", "saved_model", "pb", "tflite", "edgetpu", "tfjs", "paddle", "mnn", "ncnn", "imx", "rknn", "executorch"])
        layout.addRow("Format:", self.format_combo)

        # imgsz
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)
        layout.addRow("Image Size:", self.imgsz_spin)

        # device
        self.device_edit = QLineEdit("0")
        layout.addRow("Device:", self.device_edit)

        # batch
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1000)
        self.batch_spin.setValue(1)
        layout.addRow("Batch:", self.batch_spin)

        self.basic_tab.setLayout(layout)

    def setup_advanced_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QFormLayout(container)

        # keras
        self.keras_check = QCheckBox()
        self.keras_check.setChecked(False)
        self.keras_use = QCheckBox("Use Default")
        self.keras_use.setChecked(True)
        self.keras_check.setEnabled(False)
        self.keras_use.stateChanged.connect(lambda state: self.keras_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.keras_check)
        hbox.addWidget(self.keras_use)
        layout.addRow("Keras:", hbox)

        # optimize
        self.optimize_check = QCheckBox()
        self.optimize_check.setChecked(False)
        self.optimize_use = QCheckBox("Use Default")
        self.optimize_use.setChecked(True)
        self.optimize_check.setEnabled(False)
        self.optimize_use.stateChanged.connect(lambda state: self.optimize_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.optimize_check)
        hbox.addWidget(self.optimize_use)
        layout.addRow("Optimize:", hbox)

        # half
        self.half_check = QCheckBox()
        self.half_check.setChecked(False)
        self.half_use = QCheckBox("Use Default")
        self.half_use.setChecked(True)
        self.half_check.setEnabled(False)
        self.half_use.stateChanged.connect(lambda state: self.half_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.half_check)
        hbox.addWidget(self.half_use)
        layout.addRow("Half (FP16):", hbox)

        # int8
        self.int8_check = QCheckBox()
        self.int8_check.setChecked(False)
        self.int8_use = QCheckBox("Use Default")
        self.int8_use.setChecked(True)
        self.int8_check.setEnabled(False)
        self.int8_use.stateChanged.connect(lambda state: self.int8_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.int8_check)
        hbox.addWidget(self.int8_use)
        layout.addRow("INT8:", hbox)

        # dynamic
        self.dynamic_check = QCheckBox()
        self.dynamic_check.setChecked(False)
        self.dynamic_use = QCheckBox("Use Default")
        self.dynamic_use.setChecked(True)
        self.dynamic_check.setEnabled(False)
        self.dynamic_use.stateChanged.connect(lambda state: self.dynamic_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.dynamic_check)
        hbox.addWidget(self.dynamic_use)
        layout.addRow("Dynamic:", hbox)

        # simplify
        self.simplify_check = QCheckBox()
        self.simplify_check.setChecked(True)
        self.simplify_use = QCheckBox("Use Default")
        self.simplify_use.setChecked(True)
        self.simplify_check.setEnabled(False)
        self.simplify_use.stateChanged.connect(lambda state: self.simplify_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.simplify_check)
        hbox.addWidget(self.simplify_use)
        layout.addRow("Simplify:", hbox)

        # opset
        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(-1, 100)
        self.opset_spin.setValue(-1)  # None
        self.opset_use = QCheckBox("Use Default")
        self.opset_use.setChecked(True)
        self.opset_spin.setEnabled(False)
        self.opset_use.stateChanged.connect(lambda state: self.opset_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.opset_spin)
        hbox.addWidget(self.opset_use)
        layout.addRow("Opset:", hbox)

        # workspace
        self.workspace_spin = QDoubleSpinBox()
        self.workspace_spin.setRange(-1.0, 100.0)
        self.workspace_spin.setValue(-1.0)  # None
        self.workspace_use = QCheckBox("Use Default")
        self.workspace_use.setChecked(True)
        self.workspace_spin.setEnabled(False)
        self.workspace_use.stateChanged.connect(lambda state: self.workspace_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.workspace_spin)
        hbox.addWidget(self.workspace_use)
        layout.addRow("Workspace (GiB):", hbox)

        # nms
        self.nms_check = QCheckBox()
        self.nms_check.setChecked(False)
        self.nms_use = QCheckBox("Use Default")
        self.nms_use.setChecked(True)
        self.nms_check.setEnabled(False)
        self.nms_use.stateChanged.connect(lambda state: self.nms_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.nms_check)
        hbox.addWidget(self.nms_use)
        layout.addRow("NMS:", hbox)

        # data
        self.data_edit = QLineEdit("coco8.yaml")
        self.data_button = QPushButton("Browse")
        self.data_button.clicked.connect(self.browse_data)
        self.data_use = QCheckBox("Use Default")
        self.data_use.setChecked(True)
        self.data_edit.setEnabled(False)
        self.data_button.setEnabled(False)
        self.data_use.stateChanged.connect(lambda state: (self.data_edit.setEnabled(state == 0), self.data_button.setEnabled(state == 0)))
        hbox = QHBoxLayout()
        hbox.addWidget(self.data_edit)
        hbox.addWidget(self.data_button)
        hbox.addWidget(self.data_use)
        layout.addRow("Data:", hbox)

        # fraction
        self.fraction_spin = QDoubleSpinBox()
        self.fraction_spin.setRange(0.0, 1.0)
        self.fraction_spin.setSingleStep(0.1)
        self.fraction_spin.setValue(1.0)
        self.fraction_use = QCheckBox("Use Default")
        self.fraction_use.setChecked(True)
        self.fraction_spin.setEnabled(False)
        self.fraction_use.stateChanged.connect(lambda state: self.fraction_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.fraction_spin)
        hbox.addWidget(self.fraction_use)
        layout.addRow("Fraction:", hbox)

        scroll.setWidget(container)
        tab_layout = QVBoxLayout()
        tab_layout.addWidget(scroll)
        self.advanced_tab.setLayout(tab_layout)

    def get_args(self):
        args = {}

        # Basic
        args["model"] = self.model_edit.text()
        args["format"] = self.format_combo.currentText()
        args["imgsz"] = self.imgsz_spin.value()
        device_text = self.device_edit.text()
        if device_text.isdigit():
            args["device"] = int(device_text)
        elif device_text.startswith("[") and device_text.endswith("]"):
            args["device"] = eval(device_text)
        else:
            args["device"] = device_text
        args["batch"] = self.batch_spin.value()

        # Advanced
        if not self.keras_use.isChecked():
            args["keras"] = self.keras_check.isChecked()
        if not self.optimize_use.isChecked():
            args["optimize"] = self.optimize_check.isChecked()
        if not self.half_use.isChecked():
            args["half"] = self.half_check.isChecked()
        if not self.int8_use.isChecked():
            args["int8"] = self.int8_check.isChecked()
        if not self.dynamic_use.isChecked():
            args["dynamic"] = self.dynamic_check.isChecked()
        if not self.simplify_use.isChecked():
            args["simplify"] = self.simplify_check.isChecked()
        if not self.opset_use.isChecked():
            args["opset"] = self.opset_spin.value() if self.opset_spin.value() != -1 else None
        if not self.workspace_use.isChecked():
            args["workspace"] = self.workspace_spin.value() if self.workspace_spin.value() != -1.0 else None
        if not self.nms_use.isChecked():
            args["nms"] = self.nms_check.isChecked()
        if not self.data_use.isChecked():
            args["data"] = self.data_edit.text()
        if not self.fraction_use.isChecked():
            args["fraction"] = self.fraction_spin.value()

        return args

    def start_export(self):
        try:
            args = self.get_args()

            # Load model
            model = YOLO(args["model"])

            # Remove model from args
            args.pop("model")

            # Print export arguments
            print("\n" + "=" * 80)
            print("Export Arguments:")
            print("=" * 80)
            for key, value in sorted(args.items()):
                print(f"  {key}: {value}")
            print("=" * 80 + "\n")

            model.export(**args)
            QMessageBox.information(self, "Export Complete", "Export has finished successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")

    def export_export_command(self):
        args = self.get_args()
        model_name = args.pop("model")  # remove model from args, handle separately

        imports = ["from ultralytics import YOLO"]
        snippet = "\n".join(imports) + "\n\n"

        snippet += f'model = YOLO("{model_name}")\n'

        # Build export args
        export_args = []
        for k, v in args.items():
            if isinstance(v, str):
                export_args.append(f'{k}="{v}"')
            elif isinstance(v, bool):
                export_args.append(f"{k}={str(v)}")
            elif isinstance(v, (int, float)):
                export_args.append(f"{k}={v}")
            else:
                export_args.append(f"{k}={repr(v)}")

        if export_args:
            formatted_args = ",\n    ".join(export_args)
            snippet += "model.export(\n"
            snippet += f"    {formatted_args},\n"
            snippet += ")\n"
        else:
            snippet += "model.export()\n"

        # Create custom dialog
        from PySide6.QtWidgets import QDialog, QHBoxLayout, QTextEdit, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Export Command")
        dialog.setModal(True)

        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(snippet)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        button_layout = QHBoxLayout()

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(snippet))
        button_layout.addWidget(copy_button)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExportTab()
    window.show()
    sys.exit(app.exec())
