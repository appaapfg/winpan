import sys

import yaml
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
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
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from panopticon.widgets.about_pantrainer_dialog import AboutPanTrainerDialog


class ValidateTab(QWidget):
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

        # Visualization tab
        self.visualization_tab = QWidget()
        self.tabs.addTab(self.visualization_tab, "Visualization")
        self.setup_visualization_tab()

        # Export val command button
        self.export_button = QPushButton("Export Val Command (for Developers)")
        self.export_button.clicked.connect(self.export_val_command)

        # Validate button
        self.validate_button = QPushButton("Start Validation")
        self.validate_button.clicked.connect(self.start_validation)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.export_button)
        layout.addWidget(self.validate_button)
        self.setLayout(layout)

    def create_menu_bar(self):
        if not self.main_window:
            return
        menu_bar = self.main_window.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.main_window.close)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        load_action = edit_menu.addAction("Load Val Config")
        load_action.triggered.connect(self.load_val_config)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        save_action = edit_menu.addAction("Save Val Config")
        save_action.triggered.connect(self.save_val_config)
        save_action.setShortcut(QKeySequence.StandardKey.Save)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def show_about(self):
        dialog = AboutPanTrainerDialog(self.main_window)
        dialog.exec()

    def load_val_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Load Config", "", "PanTrainer Val Config Files (*.pan_val.yaml);;All Files (*)")
        if file_path:
            with open(file_path) as f:
                config = yaml.safe_load(f)
            self.apply_config(config)

    def save_val_config(self):
        file_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Config", "", "PanTrainer Val Config Files (*.pan_val.yaml);;All Files (*)")
        if file_path:
            config = self.collect_non_default()
            with open(file_path, "w") as f:
                yaml.dump(config, f)

    def collect_non_default(self):
        config = {}
        # Basic - always include if not default
        if self.model_edit.text() != "yolo11n.pt":
            config["model"] = self.model_edit.text()
        if self.data_edit.text() != "coco8.yaml":
            config["data"] = self.data_edit.text()
        if not self.imgsz_use.isChecked() and self.imgsz_spin.value() != 640:
            config["imgsz"] = self.imgsz_spin.value()
        if not self.batch_use.isChecked() and self.batch_spin.value() != 16:
            config["batch"] = self.batch_spin.value()
        if not self.conf_use.isChecked() and self.conf_spin.value() != 0.001:
            config["conf"] = self.conf_spin.value()
        if not self.iou_use.isChecked() and self.iou_spin.value() != 0.7:
            config["iou"] = self.iou_spin.value()
        if self.device_edit.text() != "0":
            config["device"] = self.device_edit.text()
        if not self.save_json_use.isChecked() and self.save_json_check.isChecked():
            config["save_json"] = self.save_json_check.isChecked()
        if not self.plots_use.isChecked() and not self.plots_check.isChecked():
            config["plots"] = self.plots_check.isChecked()

        # Advanced
        if not self.max_det_use.isChecked() and self.max_det_spin.value() != 300:
            config["max_det"] = self.max_det_spin.value()
        if not self.half_use.isChecked() and self.half_check.isChecked():
            config["half"] = self.half_check.isChecked()
        if not self.dnn_use.isChecked() and self.dnn_check.isChecked():
            config["dnn"] = self.dnn_check.isChecked()
        if self.split_edit.text() != "test":
            config["split"] = self.split_edit.text()
        if not self.rect_use.isChecked() and self.rect_check.isChecked():
            config["rect"] = self.rect_check.isChecked()
        if not self.workers_use.isChecked() and self.workers_spin.value() != 8:
            config["workers"] = self.workers_spin.value()
        if not self.classes_use.isChecked() and self.classes_edit.text():
            config["classes"] = self.classes_edit.text()
        if not self.agnostic_nms_use.isChecked() and self.agnostic_nms_check.isChecked():
            config["agnostic_nms"] = self.agnostic_nms_check.isChecked()
        if not self.augment_use.isChecked() and self.augment_check.isChecked():
            config["augment"] = self.augment_check.isChecked()
        if not self.visualize_use.isChecked() and self.visualize_check.isChecked():
            config["visualize"] = self.visualize_check.isChecked()

        # Visualization
        if not self.save_txt_use.isChecked() and self.save_txt_check.isChecked():
            config["save_txt"] = self.save_txt_check.isChecked()
        if not self.save_conf_use.isChecked() and self.save_conf_check.isChecked():
            config["save_conf"] = self.save_conf_check.isChecked()
        if not self.save_crop_use.isChecked() and self.save_crop_check.isChecked():
            config["save_crop"] = self.save_crop_check.isChecked()
        if not self.show_labels_use.isChecked() and not self.show_labels_check.isChecked():
            config["show_labels"] = self.show_labels_check.isChecked()
        if not self.show_conf_use.isChecked() and not self.show_conf_check.isChecked():
            config["show_conf"] = self.show_conf_check.isChecked()
        if not self.show_boxes_use.isChecked() and not self.show_boxes_check.isChecked():
            config["show_boxes"] = self.show_boxes_check.isChecked()
        if not self.line_width_use.isChecked() and self.line_width_spin.value() != 0:
            config["line_width"] = self.line_width_spin.value()

        return config

    def apply_config(self, config):
        # Basic
        if "model" in config:
            self.model_edit.setText(config["model"])
        if "data" in config:
            self.data_edit.setText(config["data"])
        if "imgsz" in config:
            self.imgsz_spin.setValue(config["imgsz"])
            self.imgsz_use.setChecked(False)
            self.imgsz_spin.setEnabled(True)
        if "batch" in config:
            self.batch_spin.setValue(config["batch"])
            self.batch_use.setChecked(False)
            self.batch_spin.setEnabled(True)
        if "conf" in config:
            self.conf_spin.setValue(config["conf"])
            self.conf_use.setChecked(False)
            self.conf_spin.setEnabled(True)
        if "iou" in config:
            self.iou_spin.setValue(config["iou"])
            self.iou_use.setChecked(False)
            self.iou_spin.setEnabled(True)
        if "device" in config:
            self.device_edit.setText(config["device"])
        if "save_json" in config:
            self.save_json_check.setChecked(config["save_json"])
            self.save_json_use.setChecked(False)
            self.save_json_check.setEnabled(True)
        if "plots" in config:
            self.plots_check.setChecked(config["plots"])
            self.plots_use.setChecked(False)
            self.plots_check.setEnabled(True)

        # Advanced
        if "max_det" in config:
            self.max_det_spin.setValue(config["max_det"])
            self.max_det_use.setChecked(False)
            self.max_det_spin.setEnabled(True)
        if "half" in config:
            self.half_check.setChecked(config["half"])
            self.half_use.setChecked(False)
            self.half_check.setEnabled(True)
        if "dnn" in config:
            self.dnn_check.setChecked(config["dnn"])
            self.dnn_use.setChecked(False)
            self.dnn_check.setEnabled(True)
        if "split" in config:
            self.split_edit.setText(config["split"])
        if "rect" in config:
            self.rect_check.setChecked(config["rect"])
            self.rect_use.setChecked(False)
            self.rect_check.setEnabled(True)
        if "workers" in config:
            self.workers_spin.setValue(config["workers"])
            self.workers_use.setChecked(False)
            self.workers_spin.setEnabled(True)
        if "classes" in config:
            self.classes_edit.setText(config["classes"])
            self.classes_use.setChecked(False)
            self.classes_edit.setEnabled(True)
        if "agnostic_nms" in config:
            self.agnostic_nms_check.setChecked(config["agnostic_nms"])
            self.agnostic_nms_use.setChecked(False)
            self.agnostic_nms_check.setEnabled(True)
        if "augment" in config:
            self.augment_check.setChecked(config["augment"])
            self.augment_use.setChecked(False)
            self.augment_check.setEnabled(True)
        if "visualize" in config:
            self.visualize_check.setChecked(config["visualize"])
            self.visualize_use.setChecked(False)
            self.visualize_check.setEnabled(True)

        # Visualization
        if "save_txt" in config:
            self.save_txt_check.setChecked(config["save_txt"])
            self.save_txt_use.setChecked(False)
            self.save_txt_check.setEnabled(True)
        if "save_conf" in config:
            self.save_conf_check.setChecked(config["save_conf"])
            self.save_conf_use.setChecked(False)
            self.save_conf_check.setEnabled(True)
        if "save_crop" in config:
            self.save_crop_check.setChecked(config["save_crop"])
            self.save_crop_use.setChecked(False)
            self.save_crop_check.setEnabled(True)
        if "show_labels" in config:
            self.show_labels_check.setChecked(config["show_labels"])
            self.show_labels_use.setChecked(False)
            self.show_labels_check.setEnabled(True)
        if "show_conf" in config:
            self.show_conf_check.setChecked(config["show_conf"])
            self.show_conf_use.setChecked(False)
            self.show_conf_check.setEnabled(True)
        if "show_boxes" in config:
            self.show_boxes_check.setChecked(config["show_boxes"])
            self.show_boxes_use.setChecked(False)
            self.show_boxes_check.setEnabled(True)
        if "line_width" in config:
            self.line_width_spin.setValue(config["line_width"])
            self.line_width_use.setChecked(False)
            self.line_width_spin.setEnabled(True)

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

        # data
        self.data_edit = QLineEdit("coco8.yaml")
        self.data_button = QPushButton("Browse")
        self.data_button.clicked.connect(self.browse_data)
        hbox = QHBoxLayout()
        hbox.addWidget(self.data_edit)
        hbox.addWidget(self.data_button)
        layout.addRow("Data:", hbox)

        # imgsz
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)
        self.imgsz_use = QCheckBox("Use Default")
        self.imgsz_use.setChecked(True)
        self.imgsz_spin.setEnabled(False)
        self.imgsz_use.stateChanged.connect(lambda state: self.imgsz_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.imgsz_spin)
        hbox.addWidget(self.imgsz_use)
        layout.addRow("Image Size:", hbox)

        # batch
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1000)
        self.batch_spin.setValue(16)
        self.batch_use = QCheckBox("Use Default")
        self.batch_use.setChecked(True)
        self.batch_spin.setEnabled(False)
        self.batch_use.stateChanged.connect(lambda state: self.batch_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.batch_spin)
        hbox.addWidget(self.batch_use)
        layout.addRow("Batch:", hbox)

        # conf
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0, 1)
        self.conf_spin.setValue(0.001)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setSingleStep(0.001)
        self.conf_use = QCheckBox("Use Default")
        self.conf_use.setChecked(True)
        self.conf_spin.setEnabled(False)
        self.conf_use.stateChanged.connect(lambda state: self.conf_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.conf_spin)
        hbox.addWidget(self.conf_use)
        layout.addRow("Confidence:", hbox)

        # iou
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0, 1)
        self.iou_spin.setValue(0.7)
        self.iou_spin.setDecimals(2)
        self.iou_use = QCheckBox("Use Default")
        self.iou_use.setChecked(True)
        self.iou_spin.setEnabled(False)
        self.iou_use.stateChanged.connect(lambda state: self.iou_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.iou_spin)
        hbox.addWidget(self.iou_use)
        layout.addRow("IoU:", hbox)

        # device
        self.device_edit = QLineEdit("0")
        layout.addRow("Device:", self.device_edit)

        # save_json
        self.save_json_check = QCheckBox()
        self.save_json_check.setChecked(False)
        self.save_json_use = QCheckBox("Use Default")
        self.save_json_use.setChecked(True)
        self.save_json_check.setEnabled(False)
        self.save_json_use.stateChanged.connect(lambda state: self.save_json_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.save_json_check)
        hbox.addWidget(self.save_json_use)
        layout.addRow("Save JSON:", hbox)

        # plots
        self.plots_check = QCheckBox()
        self.plots_check.setChecked(True)
        self.plots_use = QCheckBox("Use Default")
        self.plots_use.setChecked(True)
        self.plots_check.setEnabled(False)
        self.plots_use.stateChanged.connect(lambda state: self.plots_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.plots_check)
        hbox.addWidget(self.plots_use)
        layout.addRow("Plots:", hbox)

        self.basic_tab.setLayout(layout)

    def setup_advanced_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QFormLayout(container)

        # max_det
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 10000)
        self.max_det_spin.setValue(300)
        self.max_det_use = QCheckBox("Use Default")
        self.max_det_use.setChecked(True)
        self.max_det_spin.setEnabled(False)
        self.max_det_use.stateChanged.connect(lambda state: self.max_det_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.max_det_spin)
        hbox.addWidget(self.max_det_use)
        layout.addRow("Max Detections:", hbox)

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
        layout.addRow("Half Precision:", hbox)

        # dnn
        self.dnn_check = QCheckBox()
        self.dnn_check.setChecked(False)
        self.dnn_use = QCheckBox("Use Default")
        self.dnn_use.setChecked(True)
        self.dnn_check.setEnabled(False)
        self.dnn_use.stateChanged.connect(lambda state: self.dnn_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.dnn_check)
        hbox.addWidget(self.dnn_use)
        layout.addRow("Use DNN:", hbox)

        # split
        self.split_edit = QLineEdit("val")
        layout.addRow("Split:", self.split_edit)

        # rect
        self.rect_check = QCheckBox()
        self.rect_check.setChecked(False)
        self.rect_use = QCheckBox("Use Default")
        self.rect_use.setChecked(True)
        self.rect_check.setEnabled(False)
        self.rect_use.stateChanged.connect(lambda state: self.rect_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.rect_check)
        hbox.addWidget(self.rect_use)
        layout.addRow("Rectangular:", hbox)

        # workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 100)
        self.workers_spin.setValue(8)
        self.workers_use = QCheckBox("Use Default")
        self.workers_use.setChecked(True)
        self.workers_spin.setEnabled(False)
        self.workers_use.stateChanged.connect(lambda state: self.workers_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.workers_spin)
        hbox.addWidget(self.workers_use)
        layout.addRow("Workers:", hbox)

        # classes
        self.classes_edit = QLineEdit()
        self.classes_use = QCheckBox("Use Default")
        self.classes_use.setChecked(True)
        self.classes_edit.setEnabled(False)
        self.classes_use.stateChanged.connect(lambda state: self.classes_edit.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.classes_edit)
        hbox.addWidget(self.classes_use)
        layout.addRow("Classes:", hbox)

        # agnostic_nms
        self.agnostic_nms_check = QCheckBox()
        self.agnostic_nms_check.setChecked(False)
        self.agnostic_nms_use = QCheckBox("Use Default")
        self.agnostic_nms_use.setChecked(True)
        self.agnostic_nms_check.setEnabled(False)
        self.agnostic_nms_use.stateChanged.connect(lambda state: self.agnostic_nms_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.agnostic_nms_check)
        hbox.addWidget(self.agnostic_nms_use)
        layout.addRow("Agnostic NMS:", hbox)

        # augment
        self.augment_check = QCheckBox()
        self.augment_check.setChecked(False)
        self.augment_use = QCheckBox("Use Default")
        self.augment_use.setChecked(True)
        self.augment_check.setEnabled(False)
        self.augment_use.stateChanged.connect(lambda state: self.augment_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.augment_check)
        hbox.addWidget(self.augment_use)
        layout.addRow("Augment:", hbox)

        # visualize
        self.visualize_check = QCheckBox()
        self.visualize_check.setChecked(False)
        self.visualize_use = QCheckBox("Use Default")
        self.visualize_use.setChecked(True)
        self.visualize_check.setEnabled(False)
        self.visualize_use.stateChanged.connect(lambda state: self.visualize_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.visualize_check)
        hbox.addWidget(self.visualize_use)
        layout.addRow("Visualize:", hbox)

        scroll.setWidget(container)
        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(scroll)
        self.advanced_tab.setLayout(advanced_layout)

    def setup_visualization_tab(self):
        layout = QFormLayout()

        # save_txt
        self.save_txt_check = QCheckBox()
        self.save_txt_check.setChecked(False)
        self.save_txt_use = QCheckBox("Use Default")
        self.save_txt_use.setChecked(True)
        self.save_txt_check.setEnabled(False)
        self.save_txt_use.stateChanged.connect(lambda state: self.save_txt_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.save_txt_check)
        hbox.addWidget(self.save_txt_use)
        layout.addRow("Save TXT:", hbox)

        # save_conf
        self.save_conf_check = QCheckBox()
        self.save_conf_check.setChecked(False)
        self.save_conf_use = QCheckBox("Use Default")
        self.save_conf_use.setChecked(True)
        self.save_conf_check.setEnabled(False)
        self.save_conf_use.stateChanged.connect(lambda state: self.save_conf_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.save_conf_check)
        hbox.addWidget(self.save_conf_use)
        layout.addRow("Save Conf:", hbox)

        # save_crop
        self.save_crop_check = QCheckBox()
        self.save_crop_check.setChecked(False)
        self.save_crop_use = QCheckBox("Use Default")
        self.save_crop_use.setChecked(True)
        self.save_crop_check.setEnabled(False)
        self.save_crop_use.stateChanged.connect(lambda state: self.save_crop_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.save_crop_check)
        hbox.addWidget(self.save_crop_use)
        layout.addRow("Save Crop:", hbox)

        # show_labels
        self.show_labels_check = QCheckBox()
        self.show_labels_check.setChecked(True)
        self.show_labels_use = QCheckBox("Use Default")
        self.show_labels_use.setChecked(True)
        self.show_labels_check.setEnabled(False)
        self.show_labels_use.stateChanged.connect(lambda state: self.show_labels_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.show_labels_check)
        hbox.addWidget(self.show_labels_use)
        layout.addRow("Show Labels:", hbox)

        # show_conf
        self.show_conf_check = QCheckBox()
        self.show_conf_check.setChecked(True)
        self.show_conf_use = QCheckBox("Use Default")
        self.show_conf_use.setChecked(True)
        self.show_conf_check.setEnabled(False)
        self.show_conf_use.stateChanged.connect(lambda state: self.show_conf_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.show_conf_check)
        hbox.addWidget(self.show_conf_use)
        layout.addRow("Show Conf:", hbox)

        # show_boxes
        self.show_boxes_check = QCheckBox()
        self.show_boxes_check.setChecked(True)
        self.show_boxes_use = QCheckBox("Use Default")
        self.show_boxes_use.setChecked(True)
        self.show_boxes_check.setEnabled(False)
        self.show_boxes_use.stateChanged.connect(lambda state: self.show_boxes_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.show_boxes_check)
        hbox.addWidget(self.show_boxes_use)
        layout.addRow("Show Boxes:", hbox)

        # line_width
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(0, 100)
        self.line_width_spin.setValue(0)
        self.line_width_use = QCheckBox("Use Default")
        self.line_width_use.setChecked(True)
        self.line_width_spin.setEnabled(False)
        self.line_width_use.stateChanged.connect(lambda state: self.line_width_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.line_width_spin)
        hbox.addWidget(self.line_width_use)
        layout.addRow("Line Width:", hbox)

        self.visualization_tab.setLayout(layout)

    def get_args(self):
        args = {}

        # Basic
        args["model"] = self.model_edit.text()
        args["data"] = self.data_edit.text()
        if not self.imgsz_use.isChecked():
            args["imgsz"] = self.imgsz_spin.value()
        if not self.batch_use.isChecked():
            args["batch"] = self.batch_spin.value()
        if not self.conf_use.isChecked():
            args["conf"] = self.conf_spin.value()
        if not self.iou_use.isChecked():
            args["iou"] = self.iou_spin.value()
        device_text = self.device_edit.text()
        if device_text.isdigit():
            args["device"] = int(device_text)
        elif device_text.startswith("[") and device_text.endswith("]"):
            args["device"] = eval(device_text)
        else:
            args["device"] = device_text
        if not self.save_json_use.isChecked():
            args["save_json"] = self.save_json_check.isChecked()
        if not self.plots_use.isChecked():
            args["plots"] = self.plots_check.isChecked()

        # Advanced
        if not self.max_det_use.isChecked():
            args["max_det"] = self.max_det_spin.value()
        if not self.half_use.isChecked():
            args["half"] = self.half_check.isChecked()
        if not self.dnn_use.isChecked():
            args["dnn"] = self.dnn_check.isChecked()
        args["split"] = self.split_edit.text()
        if not self.rect_use.isChecked():
            args["rect"] = self.rect_check.isChecked()
        if not self.workers_use.isChecked():
            args["workers"] = self.workers_spin.value()
        if not self.classes_use.isChecked() and self.classes_edit.text():
            try:
                # Try to parse as list or single int
                if self.classes_edit.text().startswith("[") and self.classes_edit.text().endswith("]"):
                    args["classes"] = eval(self.classes_edit.text())
                else:
                    args["classes"] = int(self.classes_edit.text())
            except Exception:
                args["classes"] = self.classes_edit.text()
        if not self.agnostic_nms_use.isChecked():
            args["agnostic_nms"] = self.agnostic_nms_check.isChecked()
        if not self.augment_use.isChecked():
            args["augment"] = self.augment_check.isChecked()
        if not self.visualize_use.isChecked():
            args["visualize"] = self.visualize_check.isChecked()

        # Visualization
        if not self.save_txt_use.isChecked():
            args["save_txt"] = self.save_txt_check.isChecked()
        if not self.save_conf_use.isChecked():
            args["save_conf"] = self.save_conf_check.isChecked()
        if not self.save_crop_use.isChecked():
            args["save_crop"] = self.save_crop_check.isChecked()
        if not self.show_labels_use.isChecked():
            args["show_labels"] = self.show_labels_check.isChecked()
        if not self.show_conf_use.isChecked():
            args["show_conf"] = self.show_conf_check.isChecked()
        if not self.show_boxes_use.isChecked():
            args["show_boxes"] = self.show_boxes_check.isChecked()
        if not self.line_width_use.isChecked() and self.line_width_spin.value() > 0:
            args["line_width"] = self.line_width_spin.value()

        return args

    def start_validation(self):
        from ultralytics import YOLO

        args = self.get_args()
        model = YOLO(args["model"])
        try:
            results = model.val(**{k: v for k, v in args.items() if k != "model"})
            
            # Build HTML table
            html = '<html><head><style>'
            html += 'body { text-align: center; }'
            html += 'table { border-collapse: collapse; width: 80%; margin: 10px auto; }'
            html += 'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }'
            html += 'th { background-color: #000000; color: white; font-weight: bold; }'
            html += 'tr:nth-child(even) { background-color: #f9f9f9; }'
            html += 'h2 { color: #333; margin-top: 20px; }'
            html += '</style></head><body>'
            
            # Overall metrics table
            html += '<h2>Overall Metrics</h2>'
            html += '<table><tr><th>Metric</th><th>Value</th></tr>'
            
            if hasattr(results, 'box') and results.box:
                if hasattr(results.box, 'mp'):
                    html += f'<tr><td>Precision</td><td>{results.box.mp:.4f}</td></tr>'
                if hasattr(results.box, 'mr'):
                    html += f'<tr><td>Recall</td><td>{results.box.mr:.4f}</td></tr>'
                if hasattr(results.box, 'map50'):
                    html += f'<tr><td>mAP50</td><td>{results.box.map50:.4f}</td></tr>'
                if hasattr(results.box, 'map'):
                    html += f'<tr><td>mAP50-95</td><td>{results.box.map:.4f}</td></tr>'
            
            html += '</table>'
            
            # Per-class metrics table
            html += '<h2>Per-Class Metrics</h2>'
            html += '<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>mAP50</th></tr>'
            
            if hasattr(results, 'box') and results.box:
                precisions = results.box.p.tolist() if hasattr(results.box, 'p') and hasattr(results.box.p, 'tolist') else []
                recalls = results.box.r.tolist() if hasattr(results.box, 'r') and hasattr(results.box.r, 'tolist') else []
                ap50s = results.box.ap50.tolist() if hasattr(results.box, 'ap50') and hasattr(results.box.ap50, 'tolist') else []
                
                num_classes = max(len(precisions), len(recalls), len(ap50s))
                for i in range(num_classes):
                    class_name = f"Class {i}"
                    if hasattr(model, 'names') and i in model.names:
                        class_name = model.names[i]
                    
                    p = f"{precisions[i]:.4f}" if i < len(precisions) else "N/A"
                    r = f"{recalls[i]:.4f}" if i < len(recalls) else "N/A"
                    ap = f"{ap50s[i]:.4f}" if i < len(ap50s) else "N/A"
                    
                    html += f'<tr><td>{class_name}</td><td>{p}</td><td>{r}</td><td>{ap}</td></tr>'
            
            html += '</table></body></html>'
            
            # Create dialog with table
            dialog = QDialog(self.main_window)
            dialog.setWindowTitle("Validation Results")
            dialog.setGeometry(100, 100, 700, 600)
            
            layout = QVBoxLayout()
            text_browser = QTextBrowser()
            text_browser.setHtml(html)
            layout.addWidget(text_browser)
            
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(dialog.accept)
            layout.addWidget(ok_button)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Validation Error", str(e))

    def export_val_command(self):
        args = self.get_args()
        command_parts = ["from ultralytics import YOLO"]
        command_parts.append(f"model = YOLO('{args['model']}')")
        val_args = {k: v for k, v in args.items() if k != "model"}
        if val_args:
            args_str = ", ".join([f"{k}={repr(v)}" for k, v in val_args.items()])
            command_parts.append(f"results = model.val({args_str})")
        else:
            command_parts.append("results = model.val()")
        command_parts.append("print(results)")
        command = "\n".join(command_parts)
        QMessageBox.information(self.main_window, "Val Command", f"Python script:\n{command}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ValidateTab()
    window.show()
    sys.exit(app.exec())
