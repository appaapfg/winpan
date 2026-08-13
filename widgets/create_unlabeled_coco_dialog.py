import json
import os

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from panopticon.data_engine.create_ds_coco import create_unlabeled_coco_dataset


class CreateUnlabeledCocoDialog(QDialog):
    """Dialog to create an unlabeled COCO dataset for a directory of images.

    After creation, it loads the dataset into the parent PanAnnotator instance.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_annotator = parent
        self.setWindowTitle(self._tr("create_unlabeled_coco_dialog_title"))

        self.created_json_path = None

        main_layout = QVBoxLayout(self)

        description_label = QLabel(self._tr("create_unlabeled_coco_dialog_description"))
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        # Image directory selector
        img_dir_layout = QHBoxLayout()
        img_dir_layout.addWidget(QLabel(self._tr("image_directory_label")))
        self.image_dir_edit = QLineEdit()
        browse_btn = QPushButton(self._tr("browse_button"))
        browse_btn.clicked.connect(self._browse_image_dir)
        img_dir_layout.addWidget(self.image_dir_edit)
        img_dir_layout.addWidget(browse_btn)
        main_layout.addLayout(img_dir_layout)

        # Author name
        author_layout = QHBoxLayout()
        author_layout.addWidget(QLabel(self._tr("author_name_label")))
        self.author_edit = QLineEdit()
        author_layout.addWidget(self.author_edit)
        main_layout.addLayout(author_layout)

        # License name
        license_layout = QHBoxLayout()
        license_layout.addWidget(QLabel(self._tr("license_name_label")))
        self.license_edit = QLineEdit()
        license_layout.addWidget(self.license_edit)
        main_layout.addLayout(license_layout)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self._tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self._tr("cancel_button"))
        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def _tr(self, text):
        # Use parent annotator translation function if available; fallback to raw text
        if self.parent_annotator is not None and hasattr(self.parent_annotator, "tr"):
            try:
                return self.parent_annotator.tr(text)
            except Exception:
                return text
        return text

    def _browse_image_dir(self):
        d = QFileDialog.getExistingDirectory(self, self._tr("select_image_directory_dialog_title"), os.getcwd())
        if d:
            self.image_dir_edit.setText(d)

    def _on_accept(self):
        image_dir = self.image_dir_edit.text().strip()
        author = self.author_edit.text().strip()
        license_name = self.license_edit.text().strip()

        if not image_dir or not os.path.isdir(image_dir):
            QMessageBox.warning(self, self._tr("warning_title"), self._tr("invalid_image_directory_message"))
            return

        try:
            _, json_path = create_unlabeled_coco_dataset(image_dir, author, license_name)
            self.created_json_path = json_path
            self._load_into_parent(json_path)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, self._tr("error_title"), f"{self._tr('failed_create_unlabeled_coco_message')}: {e}")

    def _load_into_parent(self, json_path):
        if not self.parent_annotator:
            return
        try:
            with open(json_path) as f:
                coco_data = json.load(f)
            # Update parent annotator state directly (no modifications to its source file)
            self.parent_annotator.coco_json_path = json_path
            self.parent_annotator.coco_data = coco_data
            # Refresh category list
            if hasattr(self.parent_annotator, "class_selector") and coco_data.get("categories") is not None:
                try:
                    self.parent_annotator.class_selector.populate_categories(coco_data.get("categories", []))
                except Exception:
                    pass
            # Load images into image selector widget
            if hasattr(self.parent_annotator, "image_selector_widget"):
                self.parent_annotator.image_selector_widget.set_coco_data(coco_data, json_path)
            if hasattr(self.parent_annotator, "statusBar"):
                try:
                    self.parent_annotator.statusBar().showMessage(self._tr("status_unlabeled_coco_created_and_loaded"), 4000)
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, self._tr("error_title"), f"{self._tr('failed_load_created_coco_message')}: {e}")

    def get_created_json_path(self):
        return self.created_json_path
