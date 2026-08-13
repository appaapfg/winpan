import os

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class MergeCocoDialog(QDialog):
    """Dialog for merging multiple COCO datasets into one."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(self.parent().tr("merge_coco_dialog_title"))
        self.resize(600, 400)
        self.main_layout = QVBoxLayout(self)

        # Add a detailed description label at the top
        description_label = QLabel(self.parent().tr("merge_coco_dialog_description"))
        description_label.setWordWrap(True)
        self.main_layout.addWidget(description_label)

        layout = QVBoxLayout()

        # COCO files to merge
        layout.addWidget(QLabel(self.parent().tr("select_coco_files_label")))
        self.coco_files_list = QListWidget()
        layout.addWidget(self.coco_files_list)

        # Buttons for adding/removing files
        files_buttons_layout = QHBoxLayout()
        self.add_file_button = QPushButton(self.parent().tr("add_coco_file_button"))
        self.add_file_button.clicked.connect(self._add_coco_file)
        files_buttons_layout.addWidget(self.add_file_button)

        self.remove_file_button = QPushButton(self.parent().tr("remove_coco_file_button"))
        self.remove_file_button.clicked.connect(self._remove_selected_file)
        files_buttons_layout.addWidget(self.remove_file_button)
        layout.addLayout(files_buttons_layout)

        # Output directory
        layout.addWidget(QLabel(self.parent().tr("output_dir_label")))
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        self.browse_output_button = QPushButton(self.parent().tr("browse_button"))
        self.browse_output_button.clicked.connect(self._browse_output_dir)
        output_dir_layout.addWidget(self.browse_output_button)
        layout.addLayout(output_dir_layout)

        # Copy images checkbox
        self.copy_images_checkbox = QCheckBox(self.parent().tr("copy_images_checkbox"))
        self.copy_images_checkbox.stateChanged.connect(self._on_copy_images_toggled)
        layout.addWidget(self.copy_images_checkbox)

        # Source images directory (enabled only if copy_images is checked)
        self.source_images_layout = QHBoxLayout()
        self.source_images_label = QLabel(self.parent().tr("source_images_dir_label"))
        self.source_images_layout.addWidget(self.source_images_label)
        self.source_images_edit = QLineEdit()
        self.source_images_layout.addWidget(self.source_images_edit)
        self.browse_source_button = QPushButton(self.parent().tr("browse_source_button"))
        self.browse_source_button.clicked.connect(self._browse_source_dir)
        self.source_images_layout.addWidget(self.browse_source_button)
        layout.addLayout(self.source_images_layout)

        # Initially disable source images controls
        self._on_copy_images_toggled()

        self.main_layout.addLayout(layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.parent().tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.parent().tr("cancel_button"))
        self.main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Store the merged dataset path
        self.merged_dataset_path = None

    def _add_coco_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle(self.parent().tr("select_coco_json_files"))
        file_dialog.setNameFilter(self.parent().tr("json_files_filter"))
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for file_path in selected_files:
                # Check if already in list
                existing_items = [self.coco_files_list.item(i).text() for i in range(self.coco_files_list.count())]
                if file_path not in existing_items:
                    self.coco_files_list.addItem(QListWidgetItem(file_path))

    def _remove_selected_file(self):
        current_item = self.coco_files_list.currentItem()
        if current_item:
            row = self.coco_files_list.row(current_item)
            self.coco_files_list.takeItem(row)

    def _browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, self.parent().tr("select_output_directory"))
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _browse_source_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, self.parent().tr("select_source_images_directory"))
        if dir_path:
            self.source_images_edit.setText(dir_path)

    def _on_copy_images_toggled(self):
        enabled = self.copy_images_checkbox.isChecked()
        self.source_images_label.setEnabled(enabled)
        self.source_images_edit.setEnabled(enabled)
        self.browse_source_button.setEnabled(enabled)

    def accept(self):
        coco_files = [self.coco_files_list.item(i).text() for i in range(self.coco_files_list.count())]
        output_dir = self.output_dir_edit.text()
        copy_images = self.copy_images_checkbox.isChecked()
        source_dir = self.source_images_edit.text() if copy_images else None

        if not coco_files:
            QMessageBox.warning(self, self.parent().tr("warning"), self.parent().tr("no_files_selected"))
            return
        if not output_dir:
            QMessageBox.warning(self, self.parent().tr("warning"), self.parent().tr("output_dir_required"))
            return

        try:
            from panopticon.data_engine.merge_coco import merge_coco_datasets

            merged_path = os.path.join(output_dir, "_annotations.coco.json")
            merge_coco_datasets(coco_files, merged_path)
            if copy_images and source_dir:
                from panopticon.util.copy_coco_images import copy_coco_images

                copy_coco_images(merged_path, source_dir, output_dir)
            self.merged_dataset_path = merged_path
            QMessageBox.information(self, self.parent().tr("success"), self.parent().tr("merge_success"))
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, self.parent().tr("merge_error"), str(e))
