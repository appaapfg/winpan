import os

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from panopticon.widgets.thumbnail_viewer import ThumbnailViewer


class ImageSelectorWidget(QWidget):
    image_selected = Signal(str, object, list, int)  # path, image_obj, annotations, image_id

    def __init__(self, parent_annotator, tr_func):
        super().__init__()
        self.parent_annotator = parent_annotator
        self.tr = tr_func
        self.coco_data = None
        self.coco_images = []
        self.coco_json_path = None
        self.current_image_index = -1

        main_layout = QVBoxLayout(self)

        self.thumbnail_viewer = ThumbnailViewer(self.tr)
        main_layout.addWidget(self.thumbnail_viewer)

        navigation_layout = QHBoxLayout()

        self.prev_image_button = QPushButton(self.tr("←"))
        self.prev_image_button.clicked.connect(self.prev_image)
        self.prev_image_button.setShortcut(QKeySequence(QKeySequence.StandardKey.Back))
        navigation_layout.addWidget(self.prev_image_button)

        self.image_info_label = QLabel()
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        navigation_layout.addWidget(self.image_info_label)

        self.next_image_button = QPushButton("→")
        self.next_image_button.clicked.connect(self.next_image)
        self.next_image_button.setShortcut(QKeySequence(QKeySequence.StandardKey.Forward))
        navigation_layout.addWidget(self.next_image_button)

        main_layout.addLayout(navigation_layout)

        self.retranslate_ui()

    def set_coco_data(self, coco_data, coco_json_path):
        self.coco_data = coco_data
        self.coco_json_path = coco_json_path
        self.coco_images = self.coco_data.get("images", []) if self.coco_data else []
        self.current_image_index = -1
        self.thumbnail_viewer.set_coco_data(coco_data, coco_json_path)
        self.next_image()  # Display the first image

    def prev_image(self):
        if self.coco_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self._display_current_coco_image()

    def next_image(self):
        if self.coco_images and self.current_image_index < len(self.coco_images) - 1:
            self.current_image_index += 1
            self._display_current_coco_image()

    def _display_current_coco_image(self):
        if self.coco_images and 0 <= self.current_image_index < len(self.coco_images):
            image_info = self.coco_images[self.current_image_index]
            image_filename = image_info["file_name"]
            if self.coco_json_path:
                image_dir = os.path.dirname(self.coco_json_path)
                image_path = os.path.join(image_dir, image_filename)
            else:
                image_path = image_filename

            if os.path.exists(image_path):
                image_obj = Image.open(image_path).convert("RGB")
                annotations = []
                if self.coco_data:
                    annotations = [ann for ann in self.coco_data.get("annotations", []) if ann["image_id"] == image_info["id"]]
                self.image_selected.emit(image_path, image_obj, annotations, image_info["id"])
                self.image_info_label.setText(f"{self.current_image_index + 1} / {len(self.coco_images)}")
                self.thumbnail_viewer.update_thumbnails(self.current_image_index)
                self.parent_annotator.statusBar().showMessage(
                    self.tr("status_image_loaded").format(image_filename=image_filename),
                    2000,
                )
            else:
                QMessageBox.warning(
                    self,
                    self.tr("warning_title"),
                    self.tr("warning_image_not_found").format(image_path=image_path),
                )
        else:
            self.parent_annotator.statusBar().showMessage(self.tr("status_no_image_to_display"), 2000)

    def retranslate_ui(self):
        None

    def select_image_by_id(self, image_id: int):
        for idx, meta in enumerate(self.coco_images):
            if meta.get("id") == image_id:
                if idx != self.current_image_index:
                    self.current_image_index = idx
                    self._display_current_coco_image()
                break
