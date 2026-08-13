import os
import sys
from pathlib import Path

import yaml
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOEPETrainer

from panopticon.util.linear_probing import (
    get_backbone_freeze,
    get_backbone_neck_freeze,
    get_linear_probing_freeze,
)


class TrainTab(QWidget):
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

        # Augmentation tab
        self.augmentation_tab = QWidget()
        self.tabs.addTab(self.augmentation_tab, "Augmentation")
        self.setup_augmentation_tab()

        # Export train command button
        self.export_button = QPushButton("Export Train Command (for Developers)")
        self.export_button.clicked.connect(self.export_train_command)

        # Train button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.export_button)
        layout.addWidget(self.train_button)
        self.setLayout(layout)

    def load_train_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Load Config", "", "PanTrainer Config Files (*.pan_trainer.yaml);;All Files (*)")
        if file_path:
            if not file_path.endswith(".pan_trainer.yaml"):
                QMessageBox.warning(self, "Invalid File", "Config file must end with '.pan_trainer.yaml'")
                return
            try:
                with open(file_path) as f:
                    config = yaml.safe_load(f)
                self.apply_config(config)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")

    def save_train_config(self):
        file_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Config", "", "PanTrainer Config Files (*.pan_trainer.yaml);;All Files (*)")
        if file_path:
            if not file_path.endswith(".pan_trainer.yaml"):
                file_path += ".pan_trainer.yaml"
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
        if not self.config_use.isChecked() and self.config_edit.text():
            config["model_config"] = self.config_edit.text()
        if self.data_edit.text() != "coco8.yaml":
            config["data"] = self.data_edit.text()
        if self.epochs_spin.value() != 100:
            config["epochs"] = self.epochs_spin.value()
        if self.batch_spin.value() != 16:
            config["batch"] = self.batch_spin.value()
        if self.imgsz_spin.value() != 640:
            config["imgsz"] = self.imgsz_spin.value()
        if self.device_edit.text() != "0":
            config["device"] = self.device_edit.text()
        if not self.save_check.isChecked():
            config["save"] = self.save_check.isChecked()
        if self.project_edit.text():
            config["project"] = self.project_edit.text()
        if self.name_edit.text():
            config["name"] = self.name_edit.text()
        if self.resume_check.isChecked():
            config["resume"] = self.resume_check.isChecked()

        # notes
        if self.include_notes_check.isChecked() and self.notes_edit.toPlainText().strip():
            config["notes"] = self.notes_edit.toPlainText().strip()

        # Advanced
        if not self.time_use.isChecked():
            config["time"] = self.time_spin.value()
        if not self.patience_use.isChecked():
            config["patience"] = self.patience_spin.value()
        if not self.save_period_use.isChecked():
            config["save_period"] = self.save_period_spin.value()
        if not self.cache_use.isChecked():
            config["cache"] = self.cache_check.isChecked()
        if not self.workers_use.isChecked():
            config["workers"] = self.workers_spin.value()
        if not self.exist_ok_use.isChecked():
            config["exist_ok"] = self.exist_ok_value.isChecked()
        if not self.pretrained_use.isChecked():
            config["pretrained"] = self.pretrained_check.isChecked()
        if not self.optimizer_use.isChecked():
            config["optimizer"] = self.optimizer_edit.text()
        if not self.seed_use.isChecked():
            config["seed"] = self.seed_spin.value()
        if not self.deterministic_use.isChecked():
            config["deterministic"] = self.deterministic_value.isChecked()
        if not self.single_cls_use.isChecked():
            config["single_cls"] = self.single_cls_value.isChecked()
        if not self.classes_use.isChecked() and self.classes_edit.text():
            config["classes"] = self.classes_edit.text()
        if not self.rect_use.isChecked():
            config["rect"] = self.rect_value.isChecked()
        if not self.multi_scale_use.isChecked():
            config["multi_scale"] = self.multi_scale_value.isChecked()
        if not self.cos_lr_use.isChecked():
            config["cos_lr"] = self.cos_lr_value.isChecked()
        if not self.close_mosaic_use.isChecked():
            config["close_mosaic"] = self.close_mosaic_spin.value()
        if not self.amp_use.isChecked():
            config["amp"] = self.amp_value.isChecked()
        if not self.fraction_use.isChecked():
            config["fraction"] = self.fraction_spin.value()
        if not self.profile_use.isChecked():
            config["profile"] = self.profile_value.isChecked()
        if not self.lr0_use.isChecked():
            config["lr0"] = self.lr0_spin.value()
        if not self.lrf_use.isChecked():
            config["lrf"] = self.lrf_spin.value()
        if not self.momentum_use.isChecked():
            config["momentum"] = self.momentum_spin.value()
        if not self.weight_decay_use.isChecked():
            config["weight_decay"] = self.weight_decay_spin.value()
        if not self.warmup_epochs_use.isChecked():
            config["warmup_epochs"] = self.warmup_epochs_spin.value()
        if not self.warmup_momentum_use.isChecked():
            config["warmup_momentum"] = self.warmup_momentum_spin.value()
        if not self.warmup_bias_lr_use.isChecked():
            config["warmup_bias_lr"] = self.warmup_bias_lr_spin.value()
        if not self.box_use.isChecked():
            config["box"] = self.box_spin.value()
        if not self.cls_use.isChecked():
            config["cls"] = self.cls_spin.value()
        if not self.dfl_use.isChecked():
            config["dfl"] = self.dfl_spin.value()
        if not self.pose_use.isChecked():
            config["pose"] = self.pose_spin.value()
        if not self.kobj_use.isChecked():
            config["kobj"] = self.kobj_spin.value()
        if not self.nbs_use.isChecked():
            config["nbs"] = self.nbs_spin.value()
        if not self.overlap_mask_use.isChecked():
            config["overlap_mask"] = self.overlap_mask_value.isChecked()
        if not self.mask_ratio_use.isChecked():
            config["mask_ratio"] = self.mask_ratio_spin.value()
        if not self.dropout_use.isChecked():
            config["dropout"] = self.dropout_spin.value()
        if not self.val_use.isChecked():
            config["val"] = self.val_value.isChecked()
        if not self.plots_use.isChecked():
            config["plots"] = self.plots_value.isChecked()
        if not self.compile_use.isChecked():
            config["compile"] = self.compile_check.isChecked()
        # freeze
        if not self.freeze_use.isChecked():
            freeze_text = self.freeze_combo.currentText()
            if freeze_text == "linear probing":
                config["linear_probing"] = True
            elif freeze_text == "backbone and neck":
                config["freeze"] = 10
            elif freeze_text == "backbone":
                config["freeze"] = 12
            else:
                config["freeze"] = 0

        # trainer
        if not self.trainer_use.isChecked():
            trainer_text = self.trainer_combo.currentText()
            if trainer_text == "YOLOEPETrainer":
                config["trainer"] = "YOLOEPETrainer"
            elif trainer_text == "YOLOEPESegTrainer":
                config["trainer"] = "YOLOEPESegTrainer"

        # Augmentation
        if not self.hsv_h_use.isChecked():
            config["hsv_h"] = self.hsv_h_spin.value()
        if not self.hsv_s_use.isChecked():
            config["hsv_s"] = self.hsv_s_spin.value()
        if not self.hsv_v_use.isChecked():
            config["hsv_v"] = self.hsv_v_spin.value()
        if not self.degrees_use.isChecked():
            config["degrees"] = self.degrees_spin.value()
        if not self.translate_use.isChecked():
            config["translate"] = self.translate_spin.value()
        if not self.scale_use.isChecked():
            config["scale"] = self.scale_spin.value()
        if not self.shear_use.isChecked():
            config["shear"] = self.shear_spin.value()
        if not self.perspective_use.isChecked():
            config["perspective"] = self.perspective_spin.value()
        if not self.flipud_use.isChecked():
            config["flipud"] = self.flipud_spin.value()
        if not self.fliplr_use.isChecked():
            config["fliplr"] = self.fliplr_spin.value()
        if not self.bgr_use.isChecked():
            config["bgr"] = self.bgr_spin.value()
        if not self.mosaic_use.isChecked():
            config["mosaic"] = self.mosaic_spin.value()
        if not self.mixup_use.isChecked():
            config["mixup"] = self.mixup_spin.value()
        if not self.cutmix_use.isChecked():
            config["cutmix"] = self.cutmix_spin.value()
        if not self.copy_paste_use.isChecked():
            config["copy_paste"] = self.copy_paste_spin.value()
        if not self.copy_paste_mode_use.isChecked():
            config["copy_paste_mode"] = self.copy_paste_mode_edit.text()
        if not self.auto_augment_use.isChecked():
            config["auto_augment"] = self.auto_augment_edit.text()
        if not self.erasing_use.isChecked():
            config["erasing"] = self.erasing_spin.value()

        return config

    def apply_config(self, config):
        # Basic
        if "model" in config:
            self.model_edit.setText(config["model"])
        if "model_config" in config:
            self.config_edit.setText(config["model_config"])
            self.config_use.setChecked(False)
            self.config_edit.setEnabled(True)
            self.config_button.setEnabled(True)
        if "data" in config:
            self.data_edit.setText(config["data"])
        if "epochs" in config:
            self.epochs_spin.setValue(config["epochs"])
        if "batch" in config:
            self.batch_spin.setValue(config["batch"])
        if "imgsz" in config:
            self.imgsz_spin.setValue(config["imgsz"])
        if "device" in config:
            self.device_edit.setText(config["device"])
        if "save" in config:
            self.save_check.setChecked(config["save"])
        if "project" in config:
            self.project_edit.setText(config["project"])
        if "name" in config:
            self.name_edit.setText(config["name"])
        if "resume" in config:
            self.resume_check.setChecked(config["resume"])

        # notes
        if "notes" in config:
            self.notes_edit.setPlainText(config["notes"])
            self.include_notes_check.setChecked(True)
            self.notes_edit.setEnabled(True)
        else:
            self.include_notes_check.setChecked(False)
            self.notes_edit.setEnabled(False)

        # Advanced
        if "time" in config:
            self.time_spin.setValue(config["time"])
            self.time_use.setChecked(False)
            self.time_spin.setEnabled(True)
        if "patience" in config:
            self.patience_spin.setValue(config["patience"])
            self.patience_use.setChecked(False)
            self.patience_spin.setEnabled(True)
        if "save_period" in config:
            self.save_period_spin.setValue(config["save_period"])
            self.save_period_use.setChecked(False)
            self.save_period_spin.setEnabled(True)
        if "cache" in config:
            self.cache_check.setChecked(config["cache"])
            self.cache_use.setChecked(False)
            self.cache_check.setEnabled(True)
        if "workers" in config:
            self.workers_spin.setValue(config["workers"])
            self.workers_use.setChecked(False)
            self.workers_spin.setEnabled(True)
        if "exist_ok" in config:
            self.exist_ok_value.setChecked(config["exist_ok"])
            self.exist_ok_use.setChecked(False)
            self.exist_ok_value.setEnabled(True)
        if "pretrained" in config:
            self.pretrained_check.setChecked(config["pretrained"])
            self.pretrained_use.setChecked(False)
            self.pretrained_check.setEnabled(True)
        if "optimizer" in config:
            self.optimizer_edit.setText(config["optimizer"])
            self.optimizer_use.setChecked(False)
            self.optimizer_edit.setEnabled(True)
        if "seed" in config:
            self.seed_spin.setValue(config["seed"])
            self.seed_use.setChecked(False)
            self.seed_spin.setEnabled(True)
        if "deterministic" in config:
            self.deterministic_value.setChecked(config["deterministic"])
            self.deterministic_use.setChecked(False)
            self.deterministic_value.setEnabled(True)
        if "single_cls" in config:
            self.single_cls_value.setChecked(config["single_cls"])
            self.single_cls_use.setChecked(False)
            self.single_cls_value.setEnabled(True)
        if "classes" in config:
            self.classes_edit.setText(str(config["classes"]))
            self.classes_use.setChecked(False)
            self.classes_edit.setEnabled(True)
        if "rect" in config:
            self.rect_value.setChecked(config["rect"])
            self.rect_use.setChecked(False)
            self.rect_value.setEnabled(True)
        if "multi_scale" in config:
            self.multi_scale_value.setChecked(config["multi_scale"])
            self.multi_scale_use.setChecked(False)
            self.multi_scale_value.setEnabled(True)
        if "cos_lr" in config:
            self.cos_lr_value.setChecked(config["cos_lr"])
            self.cos_lr_use.setChecked(False)
            self.cos_lr_value.setEnabled(True)
        if "close_mosaic" in config:
            self.close_mosaic_spin.setValue(config["close_mosaic"])
            self.close_mosaic_use.setChecked(False)
            self.close_mosaic_spin.setEnabled(True)
        if "amp" in config:
            self.amp_value.setChecked(config["amp"])
            self.amp_use.setChecked(False)
            self.amp_value.setEnabled(True)
        if "fraction" in config:
            self.fraction_spin.setValue(config["fraction"])
            self.fraction_use.setChecked(False)
            self.fraction_spin.setEnabled(True)
        if "profile" in config:
            self.profile_value.setChecked(config["profile"])
            self.profile_use.setChecked(False)
            self.profile_value.setEnabled(True)
        if "lr0" in config:
            self.lr0_spin.setValue(config["lr0"])
            self.lr0_use.setChecked(False)
            self.lr0_spin.setEnabled(True)
        if "lrf" in config:
            self.lrf_spin.setValue(config["lrf"])
            self.lrf_use.setChecked(False)
            self.lrf_spin.setEnabled(True)
        if "momentum" in config:
            self.momentum_spin.setValue(config["momentum"])
            self.momentum_use.setChecked(False)
            self.momentum_spin.setEnabled(True)
        if "weight_decay" in config:
            self.weight_decay_spin.setValue(config["weight_decay"])
            self.weight_decay_use.setChecked(False)
            self.weight_decay_spin.setEnabled(True)
        if "warmup_epochs" in config:
            self.warmup_epochs_spin.setValue(config["warmup_epochs"])
            self.warmup_epochs_use.setChecked(False)
            self.warmup_epochs_spin.setEnabled(True)
        if "warmup_momentum" in config:
            self.warmup_momentum_spin.setValue(config["warmup_momentum"])
            self.warmup_momentum_use.setChecked(False)
            self.warmup_momentum_spin.setEnabled(True)
        if "warmup_bias_lr" in config:
            self.warmup_bias_lr_spin.setValue(config["warmup_bias_lr"])
            self.warmup_bias_lr_use.setChecked(False)
            self.warmup_bias_lr_spin.setEnabled(True)
        if "box" in config:
            self.box_spin.setValue(config["box"])
            self.box_use.setChecked(False)
            self.box_spin.setEnabled(True)
        if "cls" in config:
            self.cls_spin.setValue(config["cls"])
            self.cls_use.setChecked(False)
            self.cls_spin.setEnabled(True)
        if "dfl" in config:
            self.dfl_spin.setValue(config["dfl"])
            self.dfl_use.setChecked(False)
            self.dfl_spin.setEnabled(True)
        if "pose" in config:
            self.pose_spin.setValue(config["pose"])
            self.pose_use.setChecked(False)
            self.pose_spin.setEnabled(True)
        if "kobj" in config:
            self.kobj_spin.setValue(config["kobj"])
            self.kobj_use.setChecked(False)
            self.kobj_spin.setEnabled(True)
        if "nbs" in config:
            self.nbs_spin.setValue(config["nbs"])
            self.nbs_use.setChecked(False)
            self.nbs_spin.setEnabled(True)
        if "overlap_mask" in config:
            self.overlap_mask_value.setChecked(config["overlap_mask"])
            self.overlap_mask_use.setChecked(False)
            self.overlap_mask_value.setEnabled(True)
        if "mask_ratio" in config:
            self.mask_ratio_spin.setValue(config["mask_ratio"])
            self.mask_ratio_use.setChecked(False)
            self.mask_ratio_spin.setEnabled(True)
        if "dropout" in config:
            self.dropout_spin.setValue(config["dropout"])
            self.dropout_use.setChecked(False)
            self.dropout_spin.setEnabled(True)
        if "val" in config:
            self.val_value.setChecked(config["val"])
            self.val_use.setChecked(False)
            self.val_value.setEnabled(True)
        if "plots" in config:
            self.plots_value.setChecked(config["plots"])
            self.plots_use.setChecked(False)
            self.plots_value.setEnabled(True)
        if "compile" in config:
            self.compile_check.setChecked(config["compile"])
            self.compile_use.setChecked(False)
            self.compile_check.setEnabled(True)

        # freeze
        if "freeze" in config:
            freeze_val = config["freeze"]
            if freeze_val == 0:
                self.freeze_combo.setCurrentText("None")
            elif freeze_val == 10:
                self.freeze_combo.setCurrentText("backbone and neck")
            elif freeze_val == 12:
                self.freeze_combo.setCurrentText("backbone")
            self.freeze_use.setChecked(False)
            self.freeze_combo.setEnabled(True)
        elif "linear_probing" in config:
            self.freeze_combo.setCurrentText("linear probing")
            self.freeze_use.setChecked(False)
            self.freeze_combo.setEnabled(True)

        # trainer
        if "trainer" in config:
            trainer_val = config["trainer"]
            if trainer_val == "YOLOEPETrainer":
                self.trainer_combo.setCurrentText("YOLOEPETrainer")
            elif trainer_val == "YOLOEPESegTrainer":
                self.trainer_combo.setCurrentText("YOLOEPESegTrainer")
            self.trainer_use.setChecked(False)
            self.trainer_combo.setEnabled(True)

        # Augmentation
        if "hsv_h" in config:
            self.hsv_h_spin.setValue(config["hsv_h"])
            self.hsv_h_use.setChecked(False)
            self.hsv_h_spin.setEnabled(True)
        if "hsv_s" in config:
            self.hsv_s_spin.setValue(config["hsv_s"])
            self.hsv_s_use.setChecked(False)
            self.hsv_s_spin.setEnabled(True)
        if "hsv_v" in config:
            self.hsv_v_spin.setValue(config["hsv_v"])
            self.hsv_v_use.setChecked(False)
            self.hsv_v_spin.setEnabled(True)
        if "degrees" in config:
            self.degrees_spin.setValue(config["degrees"])
            self.degrees_use.setChecked(False)
            self.degrees_spin.setEnabled(True)
        if "translate" in config:
            self.translate_spin.setValue(config["translate"])
            self.translate_use.setChecked(False)
            self.translate_spin.setEnabled(True)
        if "scale" in config:
            self.scale_spin.setValue(config["scale"])
            self.scale_use.setChecked(False)
            self.scale_spin.setEnabled(True)
        if "shear" in config:
            self.shear_spin.setValue(config["shear"])
            self.shear_use.setChecked(False)
            self.shear_spin.setEnabled(True)
        if "perspective" in config:
            self.perspective_spin.setValue(config["perspective"])
            self.perspective_use.setChecked(False)
            self.perspective_spin.setEnabled(True)
        if "flipud" in config:
            self.flipud_spin.setValue(config["flipud"])
            self.flipud_use.setChecked(False)
            self.flipud_spin.setEnabled(True)
        if "fliplr" in config:
            self.fliplr_spin.setValue(config["fliplr"])
            self.fliplr_use.setChecked(False)
            self.fliplr_spin.setEnabled(True)
        if "bgr" in config:
            self.bgr_spin.setValue(config["bgr"])
            self.bgr_use.setChecked(False)
            self.bgr_spin.setEnabled(True)
        if "mosaic" in config:
            self.mosaic_spin.setValue(config["mosaic"])
            self.mosaic_use.setChecked(False)
            self.mosaic_spin.setEnabled(True)
        if "mixup" in config:
            self.mixup_spin.setValue(config["mixup"])
            self.mixup_use.setChecked(False)
            self.mixup_spin.setEnabled(True)
        if "cutmix" in config:
            self.cutmix_spin.setValue(config["cutmix"])
            self.cutmix_use.setChecked(False)
            self.cutmix_spin.setEnabled(True)
        if "copy_paste" in config:
            self.copy_paste_spin.setValue(config["copy_paste"])
            self.copy_paste_use.setChecked(False)
            self.copy_paste_spin.setEnabled(True)
        if "copy_paste_mode" in config:
            self.copy_paste_mode_edit.setText(config["copy_paste_mode"])
            self.copy_paste_mode_use.setChecked(False)
            self.copy_paste_mode_edit.setEnabled(True)
        if "auto_augment" in config:
            self.auto_augment_edit.setText(config["auto_augment"])
            self.auto_augment_use.setChecked(False)
            self.auto_augment_edit.setEnabled(True)
        if "erasing" in config:
            self.erasing_spin.setValue(config["erasing"])
            self.erasing_use.setChecked(False)
            self.erasing_spin.setEnabled(True)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Model File", "", "Model Files (*.pt *.yaml);;All Files (*)")
        if file_path:
            self.model_edit.setText(file_path)

    def browse_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Data File", "", "YAML Files (*.yaml);;All Files (*)")
        if file_path:
            self.data_edit.setText(file_path)

    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Model Config File", "", "YAML Files (*.yaml);;All Files (*)")
        if file_path:
            self.config_edit.setText(file_path)

    def setup_basic_tab(self):
        layout = QFormLayout()

        # model
        self.resume_check = QCheckBox("Resume")
        self.resume_check.setChecked(False)
        self.model_edit = QLineEdit("yolo11n.pt")
        self.model_button = QPushButton("Browse")
        self.model_button.clicked.connect(self.browse_model)
        self.resume_check.stateChanged.connect(self.on_resume_changed)
        hbox = QHBoxLayout()
        hbox.addWidget(self.resume_check)
        hbox.addWidget(self.model_edit)
        hbox.addWidget(self.model_button)
        layout.addRow("Model:", hbox)

        # model config
        self.config_edit = QLineEdit()
        self.config_button = QPushButton("Browse")
        self.config_button.clicked.connect(self.browse_config)
        self.config_use = QCheckBox("Use Default")
        self.config_use.setChecked(True)
        self.config_edit.setEnabled(False)
        self.config_button.setEnabled(False)
        self.config_use.stateChanged.connect(lambda state: (self.config_edit.setEnabled(state == 0), self.config_button.setEnabled(state == 0)))
        hbox = QHBoxLayout()
        hbox.addWidget(self.config_edit)
        hbox.addWidget(self.config_button)
        hbox.addWidget(self.config_use)
        layout.addRow("Model Config:", hbox)
        self.data_edit = QLineEdit("coco8.yaml")
        self.data_button = QPushButton("Browse")
        self.data_button.clicked.connect(self.browse_data)
        hbox = QHBoxLayout()
        hbox.addWidget(self.data_edit)
        hbox.addWidget(self.data_button)
        layout.addRow("Data:", hbox)

        # epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        layout.addRow("Epochs:", self.epochs_spin)

        # batch (use integer spinbox)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 1000)
        self.batch_spin.setValue(16)
        layout.addRow("Batch:", self.batch_spin)

        # imgsz
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setValue(640)
        layout.addRow("Image Size:", self.imgsz_spin)

        # device
        self.device_edit = QLineEdit("0")
        layout.addRow("Device:", self.device_edit)

        # save
        self.save_check = QCheckBox()
        self.save_check.setChecked(True)
        layout.addRow("Save:", self.save_check)

        # project
        self.project_edit = QLineEdit()
        layout.addRow("Project:", self.project_edit)

        # name
        self.name_edit = QLineEdit()
        layout.addRow("Name:", self.name_edit)

        # save_period
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(-1, 10000)
        self.save_period_spin.setValue(-1)
        self.save_period_use = QCheckBox("Use Default")
        self.save_period_use.setChecked(True)
        self.save_period_spin.setEnabled(False)
        self.save_period_use.stateChanged.connect(lambda state: self.save_period_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.save_period_spin)
        hbox.addWidget(self.save_period_use)
        layout.addRow("Save Period:", hbox)

        # cache
        self.cache_check = QCheckBox()
        self.cache_use = QCheckBox("Use Default")
        self.cache_use.setChecked(True)
        self.cache_check.setEnabled(False)
        self.cache_use.stateChanged.connect(lambda state: self.cache_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.cache_check)
        hbox.addWidget(self.cache_use)
        layout.addRow("Cache:", hbox)

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

        # exist_ok
        self.exist_ok_value = QCheckBox()
        self.exist_ok_use = QCheckBox("Use Default")
        self.exist_ok_use.setChecked(True)
        self.exist_ok_value.setEnabled(False)
        self.exist_ok_use.stateChanged.connect(lambda state: self.exist_ok_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.exist_ok_value)
        hbox.addWidget(self.exist_ok_use)
        layout.addRow("Exist OK:", hbox)

        # profile
        self.profile_value = QCheckBox()
        self.profile_use = QCheckBox("Use Default")
        self.profile_use.setChecked(True)
        self.profile_value.setEnabled(False)
        self.profile_use.stateChanged.connect(lambda state: self.profile_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.profile_value)
        hbox.addWidget(self.profile_use)
        layout.addRow("Profile:", hbox)

        # plots
        self.plots_value = QCheckBox()
        self.plots_use = QCheckBox("Use Default")
        self.plots_use.setChecked(True)
        self.plots_value.setEnabled(False)
        self.plots_use.stateChanged.connect(lambda state: self.plots_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.plots_value)
        hbox.addWidget(self.plots_use)
        layout.addRow("Plots:", hbox)

        # compile
        self.compile_check = QCheckBox()
        self.compile_use = QCheckBox("Use Default")
        self.compile_use.setChecked(True)
        self.compile_check.setEnabled(False)
        self.compile_use.stateChanged.connect(lambda state: self.compile_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.compile_check)
        hbox.addWidget(self.compile_use)
        layout.addRow("Compile:", hbox)

        # notes
        self.include_notes_check = QCheckBox("Include Notes")
        self.include_notes_check.setChecked(False)
        self.notes_edit = QTextEdit()
        self.notes_edit.setEnabled(False)
        self.include_notes_check.stateChanged.connect(lambda: self.notes_edit.setEnabled(self.include_notes_check.isChecked()))
        layout.addRow("Notes:", self.notes_edit)
        layout.addRow("", self.include_notes_check)

        self.basic_tab.setLayout(layout)

    def on_resume_changed(self, state):
        if state == 2:  # Checked
            self.model_edit.setEnabled(False)
            self.model_button.setEnabled(False)
            self.update_resume_model()
        else:
            self.model_edit.setEnabled(True)
            self.model_button.setEnabled(True)
            self.model_edit.setText("yolo11n.pt")

    def update_resume_model(self):
        project = self.project_edit.text().strip()
        name = self.name_edit.text().strip()
        if not project or not name:
            QMessageBox.warning(self.main_window, "Warning", "Project and Name must be set to resume training.")
            return
        runs_dir = Path(project) if project else Path("runs/train")
        if not runs_dir.exists():
            QMessageBox.warning(self.main_window, "Warning", f"Runs directory {runs_dir} does not exist.")
            return
        dirs = []

        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith(name):
                try:
                    last_modified = d.stat().st_mtime
                    dirs.append((d, last_modified))
                except Exception:
                    pass

        if not dirs:
            QMessageBox.warning(self.main_window, "Warning", f"No training directories found in {runs_dir} with name starting with '{name}'.")
            return

        dirs.sort(key=lambda x: x[1], reverse=True)

        selected_dir = dirs[0][0]
        last_pt = selected_dir / "weights" / "last.pt"
        if last_pt.exists():
            self.model_edit.setText(str(last_pt))
        else:
            QMessageBox.warning(self.main_window, "Warning", f"Last checkpoint not found in {selected_dir}.")

    def setup_advanced_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QFormLayout(container)

        # time
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0, 10000)
        self.time_use = QCheckBox("Use Default")
        self.time_use.setChecked(True)
        self.time_spin.setEnabled(False)
        self.time_use.stateChanged.connect(lambda state: self.time_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.time_spin)
        hbox.addWidget(self.time_use)
        layout.addRow("Time (hours):", hbox)

        # patience
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 10000)
        self.patience_spin.setValue(100)
        self.patience_use = QCheckBox("Use Default")
        self.patience_use.setChecked(True)
        self.patience_spin.setEnabled(False)
        self.patience_use.stateChanged.connect(lambda state: self.patience_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.patience_spin)
        hbox.addWidget(self.patience_use)
        layout.addRow("Patience:", hbox)

        # pretrained
        self.pretrained_check = QCheckBox()
        self.pretrained_check.setChecked(True)
        self.pretrained_use = QCheckBox("Use Default")
        self.pretrained_use.setChecked(True)
        self.pretrained_check.setEnabled(False)
        self.pretrained_use.stateChanged.connect(lambda state: self.pretrained_check.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.pretrained_check)
        hbox.addWidget(self.pretrained_use)
        layout.addRow("Pretrained:", hbox)

        # freeze
        self.freeze_combo = QComboBox()
        self.freeze_combo.addItem("None")
        self.freeze_combo.addItem("linear probing")
        self.freeze_combo.addItem("backbone and neck")
        self.freeze_combo.addItem("backbone")
        self.freeze_use = QCheckBox("Use Default")
        self.freeze_use.setChecked(True)
        self.freeze_combo.setEnabled(False)
        self.freeze_use.stateChanged.connect(lambda state: self.freeze_combo.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.freeze_combo)
        hbox.addWidget(self.freeze_use)
        layout.addRow("Freeze:", hbox)

        # trainer selector
        self.trainer_combo = QComboBox()
        self.trainer_combo.addItem("YOLOEPETrainer")
        self.trainer_combo.addItem("YOLOEPESegTrainer")
        self.trainer_use = QCheckBox("Use Default")
        self.trainer_use.setChecked(True)
        self.trainer_combo.setEnabled(False)
        self.trainer_use.stateChanged.connect(lambda state: self.trainer_combo.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.trainer_combo)
        hbox.addWidget(self.trainer_use)
        layout.addRow("Trainer:", hbox)

        # optimizer
        self.optimizer_edit = QLineEdit("auto")
        self.optimizer_use = QCheckBox("Use Default")
        self.optimizer_use.setChecked(True)
        self.optimizer_edit.setEnabled(False)
        self.optimizer_use.stateChanged.connect(lambda state: self.optimizer_edit.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.optimizer_edit)
        hbox.addWidget(self.optimizer_use)
        layout.addRow("Optimizer:", hbox)

        # seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1000000)
        self.seed_use = QCheckBox("Use Default")
        self.seed_use.setChecked(True)
        self.seed_spin.setEnabled(False)
        self.seed_use.stateChanged.connect(lambda state: self.seed_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.seed_spin)
        hbox.addWidget(self.seed_use)
        layout.addRow("Seed:", hbox)

        # deterministic
        self.deterministic_value = QCheckBox()
        self.deterministic_use = QCheckBox("Use Default")
        self.deterministic_use.setChecked(True)
        self.deterministic_value.setEnabled(False)
        self.deterministic_use.stateChanged.connect(lambda state: self.deterministic_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.deterministic_value)
        hbox.addWidget(self.deterministic_use)
        layout.addRow("Deterministic:", hbox)

        # single_cls
        self.single_cls_value = QCheckBox()
        self.single_cls_use = QCheckBox("Use Default")
        self.single_cls_use.setChecked(True)
        self.single_cls_value.setEnabled(False)
        self.single_cls_use.stateChanged.connect(lambda state: self.single_cls_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.single_cls_value)
        hbox.addWidget(self.single_cls_use)
        layout.addRow("Single Class:", hbox)

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

        # rect
        self.rect_value = QCheckBox()
        self.rect_use = QCheckBox("Use Default")
        self.rect_use.setChecked(True)
        self.rect_value.setEnabled(False)
        self.rect_use.stateChanged.connect(lambda state: self.rect_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.rect_value)
        hbox.addWidget(self.rect_use)
        layout.addRow("Rect:", hbox)

        # multi_scale
        self.multi_scale_value = QCheckBox()
        self.multi_scale_use = QCheckBox("Use Default")
        self.multi_scale_use.setChecked(True)
        self.multi_scale_value.setEnabled(False)
        self.multi_scale_use.stateChanged.connect(lambda state: self.multi_scale_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.multi_scale_value)
        hbox.addWidget(self.multi_scale_use)
        layout.addRow("Multi Scale:", hbox)

        # cos_lr
        self.cos_lr_value = QCheckBox()
        self.cos_lr_use = QCheckBox("Use Default")
        self.cos_lr_use.setChecked(True)
        self.cos_lr_value.setEnabled(False)
        self.cos_lr_use.stateChanged.connect(lambda state: self.cos_lr_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.cos_lr_value)
        hbox.addWidget(self.cos_lr_use)
        layout.addRow("Cos LR:", hbox)

        # close_mosaic
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(10)
        self.close_mosaic_use = QCheckBox("Use Default")
        self.close_mosaic_use.setChecked(True)
        self.close_mosaic_spin.setEnabled(False)
        self.close_mosaic_use.stateChanged.connect(lambda state: self.close_mosaic_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.close_mosaic_spin)
        hbox.addWidget(self.close_mosaic_use)
        layout.addRow("Close Mosaic:", hbox)

        # amp
        self.amp_value = QCheckBox()
        self.amp_value.setChecked(True)
        self.amp_use = QCheckBox("Use Default")
        self.amp_use.setChecked(True)
        self.amp_value.setEnabled(False)
        self.amp_use.stateChanged.connect(lambda state: self.amp_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.amp_value)
        hbox.addWidget(self.amp_use)
        layout.addRow("AMP:", hbox)

        # fraction
        self.fraction_spin = QDoubleSpinBox()
        self.fraction_spin.setRange(0, 1)
        self.fraction_spin.setValue(1.0)
        self.fraction_spin.setSingleStep(0.1)
        self.fraction_use = QCheckBox("Use Default")
        self.fraction_use.setChecked(True)
        self.fraction_spin.setEnabled(False)
        self.fraction_use.stateChanged.connect(lambda state: self.fraction_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.fraction_spin)
        hbox.addWidget(self.fraction_use)
        layout.addRow("Fraction:", hbox)

        # lr0
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0, 1)
        self.lr0_spin.setValue(0.01)
        self.lr0_spin.setDecimals(4)
        self.lr0_use = QCheckBox("Use Default")
        self.lr0_use.setChecked(True)
        self.lr0_spin.setEnabled(False)
        self.lr0_use.stateChanged.connect(lambda state: self.lr0_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.lr0_spin)
        hbox.addWidget(self.lr0_use)
        layout.addRow("LR0:", hbox)

        # lrf
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setRange(0, 1)
        self.lrf_spin.setValue(0.01)
        self.lrf_spin.setDecimals(4)
        self.lrf_use = QCheckBox("Use Default")
        self.lrf_use.setChecked(True)
        self.lrf_spin.setEnabled(False)
        self.lrf_use.stateChanged.connect(lambda state: self.lrf_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.lrf_spin)
        hbox.addWidget(self.lrf_use)
        layout.addRow("LRF:", hbox)

        # momentum
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0, 1)
        self.momentum_spin.setValue(0.937)
        self.momentum_spin.setDecimals(3)
        self.momentum_use = QCheckBox("Use Default")
        self.momentum_use.setChecked(True)
        self.momentum_spin.setEnabled(False)
        self.momentum_use.stateChanged.connect(lambda state: self.momentum_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.momentum_spin)
        hbox.addWidget(self.momentum_use)
        layout.addRow("Momentum:", hbox)

        # weight_decay
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 1)
        self.weight_decay_spin.setValue(0.0005)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_use = QCheckBox("Use Default")
        self.weight_decay_use.setChecked(True)
        self.weight_decay_spin.setEnabled(False)
        self.weight_decay_use.stateChanged.connect(lambda state: self.weight_decay_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.weight_decay_spin)
        hbox.addWidget(self.weight_decay_use)
        layout.addRow("Weight Decay:", hbox)

        # warmup_epochs
        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0, 100)
        self.warmup_epochs_spin.setValue(3.0)
        self.warmup_epochs_use = QCheckBox("Use Default")
        self.warmup_epochs_use.setChecked(True)
        self.warmup_epochs_spin.setEnabled(False)
        self.warmup_epochs_use.stateChanged.connect(lambda state: self.warmup_epochs_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.warmup_epochs_spin)
        hbox.addWidget(self.warmup_epochs_use)
        layout.addRow("Warmup Epochs:", hbox)

        # warmup_momentum
        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0, 1)
        self.warmup_momentum_spin.setValue(0.8)
        self.warmup_momentum_use = QCheckBox("Use Default")
        self.warmup_momentum_use.setChecked(True)
        self.warmup_momentum_spin.setEnabled(False)
        self.warmup_momentum_use.stateChanged.connect(lambda state: self.warmup_momentum_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.warmup_momentum_spin)
        hbox.addWidget(self.warmup_momentum_use)
        layout.addRow("Warmup Momentum:", hbox)

        # warmup_bias_lr
        self.warmup_bias_lr_spin = QDoubleSpinBox()
        self.warmup_bias_lr_spin.setRange(0, 1)
        self.warmup_bias_lr_spin.setValue(0.1)
        self.warmup_bias_lr_use = QCheckBox("Use Default")
        self.warmup_bias_lr_use.setChecked(True)
        self.warmup_bias_lr_spin.setEnabled(False)
        self.warmup_bias_lr_use.stateChanged.connect(lambda state: self.warmup_bias_lr_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.warmup_bias_lr_spin)
        hbox.addWidget(self.warmup_bias_lr_use)
        layout.addRow("Warmup Bias LR:", hbox)

        # box
        self.box_spin = QDoubleSpinBox()
        self.box_spin.setRange(0, 100)
        self.box_spin.setValue(7.5)
        self.box_use = QCheckBox("Use Default")
        self.box_use.setChecked(True)
        self.box_spin.setEnabled(False)
        self.box_use.stateChanged.connect(lambda state: self.box_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.box_spin)
        hbox.addWidget(self.box_use)
        layout.addRow("Box:", hbox)

        # cls
        self.cls_spin = QDoubleSpinBox()
        self.cls_spin.setRange(0, 100)
        self.cls_spin.setValue(0.5)
        self.cls_use = QCheckBox("Use Default")
        self.cls_use.setChecked(True)
        self.cls_spin.setEnabled(False)
        self.cls_use.stateChanged.connect(lambda state: self.cls_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.cls_spin)
        hbox.addWidget(self.cls_use)
        layout.addRow("Cls:", hbox)

        # dfl
        self.dfl_spin = QDoubleSpinBox()
        self.dfl_spin.setRange(0, 100)
        self.dfl_spin.setValue(1.5)
        self.dfl_use = QCheckBox("Use Default")
        self.dfl_use.setChecked(True)
        self.dfl_spin.setEnabled(False)
        self.dfl_use.stateChanged.connect(lambda state: self.dfl_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.dfl_spin)
        hbox.addWidget(self.dfl_use)
        layout.addRow("DFL:", hbox)

        # pose
        self.pose_spin = QDoubleSpinBox()
        self.pose_spin.setRange(0, 100)
        self.pose_spin.setValue(12.0)
        self.pose_use = QCheckBox("Use Default")
        self.pose_use.setChecked(True)
        self.pose_spin.setEnabled(False)
        self.pose_use.stateChanged.connect(lambda state: self.pose_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.pose_spin)
        hbox.addWidget(self.pose_use)
        layout.addRow("Pose:", hbox)

        # kobj
        self.kobj_spin = QDoubleSpinBox()
        self.kobj_spin.setRange(0, 100)
        self.kobj_spin.setValue(2.0)
        self.kobj_use = QCheckBox("Use Default")
        self.kobj_use.setChecked(True)
        self.kobj_spin.setEnabled(False)
        self.kobj_use.stateChanged.connect(lambda state: self.kobj_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.kobj_spin)
        hbox.addWidget(self.kobj_use)
        layout.addRow("KObj:", hbox)

        # nbs
        self.nbs_spin = QSpinBox()
        self.nbs_spin.setRange(1, 10000)
        self.nbs_spin.setValue(64)
        self.nbs_use = QCheckBox("Use Default")
        self.nbs_use.setChecked(True)
        self.nbs_spin.setEnabled(False)
        self.nbs_use.stateChanged.connect(lambda state: self.nbs_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.nbs_spin)
        hbox.addWidget(self.nbs_use)
        layout.addRow("NBS:", hbox)

        # overlap_mask
        self.overlap_mask_value = QCheckBox()
        self.overlap_mask_value.setChecked(True)
        self.overlap_mask_use = QCheckBox("Use Default")
        self.overlap_mask_use.setChecked(True)
        self.overlap_mask_value.setEnabled(False)
        self.overlap_mask_use.stateChanged.connect(lambda state: self.overlap_mask_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.overlap_mask_value)
        hbox.addWidget(self.overlap_mask_use)
        layout.addRow("Overlap Mask:", hbox)

        # mask_ratio
        self.mask_ratio_spin = QSpinBox()
        self.mask_ratio_spin.setRange(1, 100)
        self.mask_ratio_spin.setValue(4)
        self.mask_ratio_use = QCheckBox("Use Default")
        self.mask_ratio_use.setChecked(True)
        self.mask_ratio_spin.setEnabled(False)
        self.mask_ratio_use.stateChanged.connect(lambda state: self.mask_ratio_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.mask_ratio_spin)
        hbox.addWidget(self.mask_ratio_use)
        layout.addRow("Mask Ratio:", hbox)

        # dropout
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0, 1)
        self.dropout_use = QCheckBox("Use Default")
        self.dropout_use.setChecked(True)
        self.dropout_spin.setEnabled(False)
        self.dropout_use.stateChanged.connect(lambda state: self.dropout_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.dropout_spin)
        hbox.addWidget(self.dropout_use)
        layout.addRow("Dropout:", hbox)

        # val
        self.val_value = QCheckBox()
        self.val_value.setChecked(True)
        self.val_use = QCheckBox("Use Default")
        self.val_use.setChecked(True)
        self.val_value.setEnabled(False)
        self.val_use.stateChanged.connect(lambda state: self.val_value.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.val_value)
        hbox.addWidget(self.val_use)
        layout.addRow("Val:", hbox)

        scroll.setWidget(container)
        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(scroll)
        self.advanced_tab.setLayout(advanced_layout)

    def setup_augmentation_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QFormLayout(container)

        # hsv_h
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0, 1)
        self.hsv_h_spin.setValue(0.015)
        self.hsv_h_spin.setSingleStep(0.001)
        self.hsv_h_use = QCheckBox("Use Default")
        self.hsv_h_use.setChecked(True)
        self.hsv_h_spin.setEnabled(False)
        self.hsv_h_use.stateChanged.connect(lambda state: self.hsv_h_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.hsv_h_spin)
        hbox.addWidget(self.hsv_h_use)
        layout.addRow("HSV H:", hbox)

        # hsv_s
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0, 1)
        self.hsv_s_spin.setValue(0.7)
        self.hsv_s_use = QCheckBox("Use Default")
        self.hsv_s_use.setChecked(True)
        self.hsv_s_spin.setEnabled(False)
        self.hsv_s_use.stateChanged.connect(lambda state: self.hsv_s_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.hsv_s_spin)
        hbox.addWidget(self.hsv_s_use)
        layout.addRow("HSV S:", hbox)

        # hsv_v
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0, 1)
        self.hsv_v_spin.setValue(0.4)
        self.hsv_v_use = QCheckBox("Use Default")
        self.hsv_v_use.setChecked(True)
        self.hsv_v_spin.setEnabled(False)
        self.hsv_v_use.stateChanged.connect(lambda state: self.hsv_v_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.hsv_v_spin)
        hbox.addWidget(self.hsv_v_use)
        layout.addRow("HSV V:", hbox)

        # degrees
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0, 180)
        self.degrees_use = QCheckBox("Use Default")
        self.degrees_use.setChecked(True)
        self.degrees_spin.setEnabled(False)
        self.degrees_use.stateChanged.connect(lambda state: self.degrees_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.degrees_spin)
        hbox.addWidget(self.degrees_use)
        layout.addRow("Degrees:", hbox)

        # translate
        self.translate_spin = QDoubleSpinBox()
        self.translate_spin.setRange(0, 1)
        self.translate_spin.setValue(0.1)
        self.translate_use = QCheckBox("Use Default")
        self.translate_use.setChecked(True)
        self.translate_spin.setEnabled(False)
        self.translate_use.stateChanged.connect(lambda state: self.translate_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.translate_spin)
        hbox.addWidget(self.translate_use)
        layout.addRow("Translate:", hbox)

        # scale
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0, 10)
        self.scale_spin.setValue(0.5)
        self.scale_use = QCheckBox("Use Default")
        self.scale_use.setChecked(True)
        self.scale_spin.setEnabled(False)
        self.scale_use.stateChanged.connect(lambda state: self.scale_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.scale_spin)
        hbox.addWidget(self.scale_use)
        layout.addRow("Scale:", hbox)

        # shear
        self.shear_spin = QDoubleSpinBox()
        self.shear_spin.setRange(-180, 180)
        self.shear_use = QCheckBox("Use Default")
        self.shear_use.setChecked(True)
        self.shear_spin.setEnabled(False)
        self.shear_use.stateChanged.connect(lambda state: self.shear_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.shear_spin)
        hbox.addWidget(self.shear_use)
        layout.addRow("Shear:", hbox)

        # perspective
        self.perspective_spin = QDoubleSpinBox()
        self.perspective_spin.setRange(0, 0.001)
        self.perspective_spin.setSingleStep(0.00001)
        self.perspective_use = QCheckBox("Use Default")
        self.perspective_use.setChecked(True)
        self.perspective_spin.setEnabled(False)
        self.perspective_use.stateChanged.connect(lambda state: self.perspective_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.perspective_spin)
        hbox.addWidget(self.perspective_use)
        layout.addRow("Perspective:", hbox)

        # flipud
        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0, 1)
        self.flipud_use = QCheckBox("Use Default")
        self.flipud_use.setChecked(True)
        self.flipud_spin.setEnabled(False)
        self.flipud_use.stateChanged.connect(lambda state: self.flipud_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.flipud_spin)
        hbox.addWidget(self.flipud_use)
        layout.addRow("Flip UD:", hbox)

        # fliplr
        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0, 1)
        self.fliplr_spin.setValue(0.5)
        self.fliplr_use = QCheckBox("Use Default")
        self.fliplr_use.setChecked(True)
        self.fliplr_spin.setEnabled(False)
        self.fliplr_use.stateChanged.connect(lambda state: self.fliplr_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.fliplr_spin)
        hbox.addWidget(self.fliplr_use)
        layout.addRow("Flip LR:", hbox)

        # bgr
        self.bgr_spin = QDoubleSpinBox()
        self.bgr_spin.setRange(0, 1)
        self.bgr_use = QCheckBox("Use Default")
        self.bgr_use.setChecked(True)
        self.bgr_spin.setEnabled(False)
        self.bgr_use.stateChanged.connect(lambda state: self.bgr_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.bgr_spin)
        hbox.addWidget(self.bgr_use)
        layout.addRow("BGR:", hbox)

        # mosaic
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0, 1)
        self.mosaic_spin.setValue(1.0)
        self.mosaic_use = QCheckBox("Use Default")
        self.mosaic_use.setChecked(True)
        self.mosaic_spin.setEnabled(False)
        self.mosaic_use.stateChanged.connect(lambda state: self.mosaic_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.mosaic_spin)
        hbox.addWidget(self.mosaic_use)
        layout.addRow("Mosaic:", hbox)

        # mixup
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0, 1)
        self.mixup_use = QCheckBox("Use Default")
        self.mixup_use.setChecked(True)
        self.mixup_spin.setEnabled(False)
        self.mixup_use.stateChanged.connect(lambda state: self.mixup_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.mixup_spin)
        hbox.addWidget(self.mixup_use)
        layout.addRow("Mixup:", hbox)

        # cutmix
        self.cutmix_spin = QDoubleSpinBox()
        self.cutmix_spin.setRange(0, 1)
        self.cutmix_use = QCheckBox("Use Default")
        self.cutmix_use.setChecked(True)
        self.cutmix_spin.setEnabled(False)
        self.cutmix_use.stateChanged.connect(lambda state: self.cutmix_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.cutmix_spin)
        hbox.addWidget(self.cutmix_use)
        layout.addRow("CutMix:", hbox)

        # copy_paste
        self.copy_paste_spin = QDoubleSpinBox()
        self.copy_paste_spin.setRange(0, 1)
        self.copy_paste_use = QCheckBox("Use Default")
        self.copy_paste_use.setChecked(True)
        self.copy_paste_spin.setEnabled(False)
        self.copy_paste_use.stateChanged.connect(lambda state: self.copy_paste_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.copy_paste_spin)
        hbox.addWidget(self.copy_paste_use)
        layout.addRow("Copy Paste:", hbox)

        # copy_paste_mode
        self.copy_paste_mode_edit = QLineEdit("flip")
        self.copy_paste_mode_use = QCheckBox("Use Default")
        self.copy_paste_mode_use.setChecked(True)
        self.copy_paste_mode_edit.setEnabled(False)
        self.copy_paste_mode_use.stateChanged.connect(lambda state: self.copy_paste_mode_edit.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.copy_paste_mode_edit)
        hbox.addWidget(self.copy_paste_mode_use)
        layout.addRow("Copy Paste Mode:", hbox)

        # auto_augment
        self.auto_augment_edit = QLineEdit("randaugment")
        self.auto_augment_use = QCheckBox("Use Default")
        self.auto_augment_use.setChecked(True)
        self.auto_augment_edit.setEnabled(False)
        self.auto_augment_use.stateChanged.connect(lambda state: self.auto_augment_edit.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.auto_augment_edit)
        hbox.addWidget(self.auto_augment_use)
        layout.addRow("Auto Augment:", hbox)

        # erasing
        self.erasing_spin = QDoubleSpinBox()
        self.erasing_spin.setRange(0, 0.9)
        self.erasing_spin.setValue(0.4)
        self.erasing_use = QCheckBox("Use Default")
        self.erasing_use.setChecked(True)
        self.erasing_spin.setEnabled(False)
        self.erasing_use.stateChanged.connect(lambda state: self.erasing_spin.setEnabled(state == 0))
        hbox = QHBoxLayout()
        hbox.addWidget(self.erasing_spin)
        hbox.addWidget(self.erasing_use)
        layout.addRow("Erasing:", hbox)

        scroll.setWidget(container)
        aug_layout = QVBoxLayout()
        aug_layout.addWidget(scroll)
        self.augmentation_tab.setLayout(aug_layout)

    def get_args(self):
        args = {}

        # Basic
        args["model"] = self.model_edit.text()
        if not self.config_use.isChecked() and self.config_edit.text():
            args["model_config"] = self.config_edit.text()
        args["data"] = self.data_edit.text()
        args["epochs"] = self.epochs_spin.value()
        args["batch"] = self.batch_spin.value()
        args["imgsz"] = self.imgsz_spin.value()
        device_text = self.device_edit.text()
        if device_text.isdigit():
            args["device"] = int(device_text)
        elif device_text.startswith("[") and device_text.endswith("]"):
            args["device"] = eval(device_text)
        else:
            args["device"] = device_text
        args["save"] = self.save_check.isChecked()
        if self.project_edit.text():
            args["project"] = self.project_edit.text()
        if self.name_edit.text():
            args["name"] = self.name_edit.text()
        if self.resume_check.isChecked():
            args["resume"] = self.resume_check.isChecked()

        # Advanced
        if not self.time_use.isChecked():
            args["time"] = self.time_spin.value()
        if not self.patience_use.isChecked():
            args["patience"] = self.patience_spin.value()
        if not self.save_period_use.isChecked():
            args["save_period"] = self.save_period_spin.value()
        if not self.cache_use.isChecked():
            args["cache"] = self.cache_check.isChecked()
        if not self.workers_use.isChecked():
            args["workers"] = self.workers_spin.value()
        if not self.exist_ok_use.isChecked():
            args["exist_ok"] = self.exist_ok_value.isChecked()
        args["pretrained"] = self.pretrained_check.isChecked()
        if not self.optimizer_use.isChecked():
            args["optimizer"] = self.optimizer_edit.text()
        if not self.seed_use.isChecked():
            args["seed"] = self.seed_spin.value()
        if not self.deterministic_use.isChecked():
            args["deterministic"] = self.deterministic_value.isChecked()
        if not self.single_cls_use.isChecked():
            args["single_cls"] = self.single_cls_value.isChecked()
        if not self.classes_use.isChecked() and self.classes_edit.text():
            classes_text = self.classes_edit.text().strip()
            try:
                if classes_text.startswith("[") and classes_text.endswith("]"):
                    args["classes"] = eval(classes_text)
                else:
                    args["classes"] = [int(x) for x in classes_text.replace(",", " ").split()]
            except Exception:
                args["classes"] = classes_text
        if not self.rect_use.isChecked():
            args["rect"] = self.rect_value.isChecked()
        if not self.multi_scale_use.isChecked():
            args["multi_scale"] = self.multi_scale_value.isChecked()
        if not self.cos_lr_use.isChecked():
            args["cos_lr"] = self.cos_lr_value.isChecked()
        if not self.close_mosaic_use.isChecked():
            args["close_mosaic"] = self.close_mosaic_spin.value()
        if not self.amp_use.isChecked():
            args["amp"] = self.amp_value.isChecked()
        if not self.fraction_use.isChecked():
            args["fraction"] = self.fraction_spin.value()
        if not self.profile_use.isChecked():
            args["profile"] = self.profile_value.isChecked()
        if not self.lr0_use.isChecked():
            args["lr0"] = self.lr0_spin.value()
        if not self.lrf_use.isChecked():
            args["lrf"] = self.lrf_spin.value()
        if not self.momentum_use.isChecked():
            args["momentum"] = self.momentum_spin.value()
        if not self.weight_decay_use.isChecked():
            args["weight_decay"] = self.weight_decay_spin.value()
        if not self.warmup_epochs_use.isChecked():
            args["warmup_epochs"] = self.warmup_epochs_spin.value()
        if not self.warmup_momentum_use.isChecked():
            args["warmup_momentum"] = self.warmup_momentum_spin.value()
        if not self.warmup_bias_lr_use.isChecked():
            args["warmup_bias_lr"] = self.warmup_bias_lr_spin.value()
        if not self.box_use.isChecked():
            args["box"] = self.box_spin.value()
        if not self.cls_use.isChecked():
            args["cls"] = self.cls_spin.value()
        if not self.dfl_use.isChecked():
            args["dfl"] = self.dfl_spin.value()
        if not self.pose_use.isChecked():
            args["pose"] = self.pose_spin.value()
        if not self.kobj_use.isChecked():
            args["kobj"] = self.kobj_spin.value()
        if not self.nbs_use.isChecked():
            args["nbs"] = self.nbs_spin.value()
        if not self.overlap_mask_use.isChecked():
            args["overlap_mask"] = self.overlap_mask_value.isChecked()
        if not self.mask_ratio_use.isChecked():
            args["mask_ratio"] = self.mask_ratio_spin.value()
        if not self.dropout_use.isChecked():
            args["dropout"] = self.dropout_spin.value()
        if not self.val_use.isChecked():
            args["val"] = self.val_value.isChecked()
        if not self.plots_use.isChecked():
            args["plots"] = self.plots_value.isChecked()
        if not self.compile_use.isChecked():
            args["compile"] = self.compile_check.isChecked()
        # freeze
        if not self.freeze_use.isChecked():
            freeze_text = self.freeze_combo.currentText()
            if freeze_text == "linear probing":
                args["freeze"] = get_linear_probing_freeze(self.model_edit.text())
            elif freeze_text == "backbone and neck":
                args["freeze"] = 10
            elif freeze_text == "backbone":
                args["freeze"] = 12
            else:
                args["freeze"] = 0

        # trainer
        if not self.trainer_use.isChecked():
            trainer_text = self.trainer_combo.currentText()
            if trainer_text == "YOLOEPETrainer":
                args["trainer"] = YOLOEPETrainer
            elif trainer_text == "YOLOEPESegTrainer":
                args["trainer"] = YOLOEPESegTrainer

        # Augmentation
        if not self.hsv_h_use.isChecked():
            args["hsv_h"] = self.hsv_h_spin.value()
        if not self.hsv_s_use.isChecked():
            args["hsv_s"] = self.hsv_s_spin.value()
        if not self.hsv_v_use.isChecked():
            args["hsv_v"] = self.hsv_v_spin.value()
        if not self.degrees_use.isChecked():
            args["degrees"] = self.degrees_spin.value()
        if not self.translate_use.isChecked():
            args["translate"] = self.translate_spin.value()
        if not self.scale_use.isChecked():
            args["scale"] = self.scale_spin.value()
        if not self.shear_use.isChecked():
            args["shear"] = self.shear_spin.value()
        if not self.perspective_use.isChecked():
            args["perspective"] = self.perspective_spin.value()
        if not self.flipud_use.isChecked():
            args["flipud"] = self.flipud_spin.value()
        if not self.fliplr_use.isChecked():
            args["fliplr"] = self.fliplr_spin.value()
        if not self.bgr_use.isChecked():
            args["bgr"] = self.bgr_spin.value()
        if not self.mosaic_use.isChecked():
            args["mosaic"] = self.mosaic_spin.value()
        if not self.mixup_use.isChecked():
            args["mixup"] = self.mixup_spin.value()
        if not self.cutmix_use.isChecked():
            args["cutmix"] = self.cutmix_spin.value()
        if not self.copy_paste_use.isChecked():
            args["copy_paste"] = self.copy_paste_spin.value()
        if not self.copy_paste_mode_use.isChecked():
            args["copy_paste_mode"] = self.copy_paste_mode_edit.text()
        if not self.auto_augment_use.isChecked():
            args["auto_augment"] = self.auto_augment_edit.text()
        if not self.erasing_use.isChecked():
            args["erasing"] = self.erasing_spin.value()

        return args

    def start_training(self):
        try:
            args = self.get_args()

            # Load model and train
            if args["model"].startswith("yoloe"):
                if "model_config" in args:
                    model = YOLOE(args["model_config"]).load(args["model"])
                else:
                    model = YOLOE(args["model"])
            else:
                if "model_config" in args:
                    model = YOLO(args["model_config"]).load(args["model"])
                else:
                    model = YOLO(args["model"])
            # Remove model_config from args as it's only used for initialization
            args.pop("model_config", None)
            # If freeze requested, compute freeze list and pass to train
            if "freeze" in args and isinstance(args["freeze"], str):
                freeze_mode = args["freeze"]
                if freeze_mode == "linear probing":
                    try:
                        freeze_list = get_linear_probing_freeze(model)
                        args["freeze"] = freeze_list
                    except Exception as e:
                        QMessageBox.warning(self, "Freeze", f"Failed to compute linear probing freeze list: {e}\nProceeding without freeze.")
                        args.pop("freeze", None)
                elif freeze_mode == "backbone and neck":
                    try:
                        freeze_list = get_backbone_neck_freeze(model)
                        args["freeze"] = freeze_list
                    except Exception as e:
                        QMessageBox.warning(self, "Freeze", f"Failed to compute backbone and neck freeze list: {e}\nProceeding without freeze.")
                        args.pop("freeze", None)
                elif freeze_mode == "backbone":
                    try:
                        freeze_list = get_backbone_freeze(model)
                        args["freeze"] = freeze_list
                    except Exception as e:
                        QMessageBox.warning(self, "Freeze", f"Failed to compute backbone freeze list: {e}\nProceeding without freeze.")
                        args.pop("freeze", None)

            # Print training arguments
            print("\n" + "=" * 80)
            print("Training Arguments:")
            print("=" * 80)
            for key, value in sorted(args.items()):
                print(f"  {key}: {value}")
            print("=" * 80 + "\n")

            model.train(**args)
            QMessageBox.information(self, "Training Complete", "Training has finished successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n{str(e)}")

    def export_train_command(self):
        args = self.get_args()
        model_name = args.pop("model")  # remove model from args, handle separately

        imports = ["from ultralytics import YOLO, YOLOE"]
        if "trainer" in args and isinstance(args["trainer"], type):
            imports.append("from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOEPETrainer")

        snippet = "\n".join(imports) + "\n\n"

        if model_name.startswith("yoloe"):
            if "model_config" in args:
                config_name = args.pop("model_config")
                snippet += f'model = YOLOE("{config_name}").load("{model_name}")\n'
            else:
                snippet += f'model = YOLOE("{model_name}")\n'
        else:
            if "model_config" in args:
                config_name = args.pop("model_config")
                snippet += f'model = YOLO("{config_name}").load("{model_name}")\n'
            else:
                snippet += f'model = YOLO("{model_name}")\n'

        # Freeze options (include linear_probing.py if any freeze option is used)
        freeze_mode = args.get("freeze")
        if freeze_mode and freeze_mode != "None":
            lp_path = os.path.join(os.path.dirname(__file__), "util", "linear_probing.py")
            with open(lp_path) as _f:
                lp_contents = _f.read()

            snippet += lp_contents

        # Build train args
        train_args = []
        for k, v in args.items():
            if k == "freeze" and isinstance(v, str):
                # Handle freeze option - convert to function call
                if v == "linear probing":
                    train_args.append("freeze=get_linear_probing_freeze(model)")
                elif v == "backbone and neck":
                    train_args.append("freeze=get_backbone_neck_freeze(model)")
                elif v == "backbone":
                    train_args.append("freeze=get_backbone_freeze(model)")
                continue
            elif isinstance(v, str):
                train_args.append(f'{k}="{v}"')
            elif isinstance(v, bool):
                train_args.append(f"{k}={str(v)}")
            elif isinstance(v, (int, float)):
                train_args.append(f"{k}={v}")
            elif isinstance(v, list):
                train_args.append(f"{k}={v}")
            elif k == "trainer" and isinstance(v, type):
                if v == YOLOEPETrainer:
                    train_args.append("trainer=YOLOEPETrainer")
                elif v == YOLOEPESegTrainer:
                    train_args.append("trainer=YOLOEPESegTrainer")
                else:
                    train_args.append(f"trainer={v.__name__}")
            else:
                train_args.append(f"{k}={repr(v)}")

        if train_args:
            formatted_args = ",\n    ".join(train_args)
            snippet += "model.train(\n"
            snippet += f"    {formatted_args},\n"
            snippet += ")\n"
        else:
            snippet += "model.train()\n"

        # Create custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Train Command")
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
    window = TrainTab()
    window.show()
    sys.exit(app.exec())
