import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from panopticon.data_engine.merge_coco import merge_coco_datasets
from panopticon.util.copy_coco_images import copy_coco_images


class MergeDatasetsDialog(QDialog):
    """Dialog for merging multiple COCO datasets into one.

    Args to gather:
        coco_files: list[str] - paths to COCO JSON files
        output_folder: str - folder to save merged dataset
        copy_images: bool - whether to copy images
        source_folder: str - folder containing images (if copy_images)
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(self.parent().tr("merge_datasets_dialog_title"))
        self.resize(800, 600)
        self.main_layout = QVBoxLayout(self)

        # Create a scroll area for the main content
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)
        self.scroll_area.setWidget(self.scroll_area_content)
        self.main_layout.addWidget(self.scroll_area)

        # Add a detailed description label at the top
        description_label = QLabel(self.parent().tr("merge_datasets_dialog_description"))
        description_label.setWordWrap(True)
        self.scroll_area_layout.addWidget(description_label)

        # Select COCO JSON files
        self.scroll_area_layout.addWidget(QLabel(self.parent().tr("select_coco_files_label")))
        coco_files_layout = QHBoxLayout()
        self.coco_files_list = QListWidget()
        self.coco_files_list.setMaximumHeight(100)
        coco_files_layout.addWidget(self.coco_files_list)
        coco_files_buttons_layout = QVBoxLayout()
        self.add_coco_files_button = QPushButton(self.parent().tr("add_coco_files_button"))
        self.add_coco_files_button.clicked.connect(self._add_coco_files)
        coco_files_buttons_layout.addWidget(self.add_coco_files_button)
        self.remove_coco_file_button = QPushButton(self.parent().tr("remove_coco_file_button"))
        self.remove_coco_file_button.clicked.connect(self._remove_coco_file)
        coco_files_buttons_layout.addWidget(self.remove_coco_file_button)
        coco_files_layout.addLayout(coco_files_buttons_layout)
        self.scroll_area_layout.addLayout(coco_files_layout)

        # Output folder
        self.scroll_area_layout.addWidget(QLabel(self.parent().tr("output_folder_label")))
        output_folder_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        output_folder_layout.addWidget(self.output_folder_edit)
        self.browse_output_button = QPushButton(self.parent().tr("browse_button"))
        self.browse_output_button.clicked.connect(self._browse_output_folder)
        output_folder_layout.addWidget(self.browse_output_button)
        self.scroll_area_layout.addLayout(output_folder_layout)

        # Copy images checkbox
        self.copy_images_checkbox = QCheckBox(self.parent().tr("copy_images_checkbox"))
        self.copy_images_checkbox.stateChanged.connect(self._on_copy_images_changed)
        self.scroll_area_layout.addWidget(self.copy_images_checkbox)

        # Source folder for images
        self.source_folder_label = QLabel(self.parent().tr("source_folder_label"))
        self.scroll_area_layout.addWidget(self.source_folder_label)
        source_folder_layout = QHBoxLayout()
        self.source_folder_edit = QLineEdit()
        source_folder_layout.addWidget(self.source_folder_edit)
        self.browse_source_button = QPushButton(self.parent().tr("browse_button"))
        self.browse_source_button.clicked.connect(self._browse_source_folder)
        source_folder_layout.addWidget(self.browse_source_button)
        self.scroll_area_layout.addLayout(source_folder_layout)

        # Initially disable source folder
        self._on_copy_images_changed(Qt.CheckState.Unchecked)

        # Button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.parent().tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.parent().tr("cancel_button"))
        self.main_layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _add_coco_files(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle(self.parent().tr("select_coco_files"))
        file_dialog.setNameFilter(self.parent().tr("json_files_filter"))
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for file_path in selected_files:
                if file_path not in [self.coco_files_list.item(i).text() for i in range(self.coco_files_list.count())]:
                    self.coco_files_list.addItem(file_path)

    def _remove_coco_file(self):
        current_item = self.coco_files_list.currentItem()
        if current_item:
            self.coco_files_list.takeItem(self.coco_files_list.row(current_item))

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.parent().tr("select_output_folder"))
        if folder:
            self.output_folder_edit.setText(folder)

    def _browse_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.parent().tr("select_source_folder"))
        if folder:
            self.source_folder_edit.setText(folder)

    def _on_copy_images_changed(self, state):
        enabled = state == Qt.CheckState.Checked
        self.source_folder_label.setEnabled(enabled)
        self.source_folder_edit.setEnabled(enabled)
        self.browse_source_button.setEnabled(enabled)

    def accept(self):
        coco_files = [self.coco_files_list.item(i).text() for i in range(self.coco_files_list.count())]
        if not coco_files:
            QMessageBox.warning(self, self.parent().tr("warning_title"), self.parent().tr("no_coco_files_selected"))
            return
        output_folder = self.output_folder_edit.text().strip()
        if not output_folder:
            QMessageBox.warning(self, self.parent().tr("warning_title"), self.parent().tr("no_output_folder_selected"))
            return
        if not os.path.isdir(output_folder):
            QMessageBox.warning(self, self.parent().tr("warning_title"), self.parent().tr("output_folder_not_exist"))
            return
        copy_images = self.copy_images_checkbox.isChecked()
        source_folder = self.source_folder_edit.text().strip() if copy_images else None
        if copy_images and not source_folder:
            QMessageBox.warning(self, self.parent().tr("warning_title"), self.parent().tr("no_source_folder_selected"))
            return
        if copy_images and not os.path.isdir(source_folder):
            QMessageBox.warning(self, self.parent().tr("warning_title"), self.parent().tr("source_folder_not_exist"))
            return

        try:
            merged = merge_coco_datasets(coco_files, output_file=os.path.join(output_folder, 'merged.coco.json'))
            if copy_images:
                copy_coco_images(os.path.join(output_folder, 'merged.coco.json'), source_folder, output_folder)
            QMessageBox.information(self, self.parent().tr("success_title"), self.parent().tr("merge_success_message").format(
                images=len(merged['images']), annotations=len(merged['annotations']), categories=len(merged['categories'])
            ))
        except ValueError as e:
            QMessageBox.critical(self, self.parent().tr("error_title"), str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, self.parent().tr("error_title"), self.parent().tr("merge_error_message").format(error=str(e)))
            return

        super().accept()
