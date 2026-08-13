import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget


class ThumbnailViewer(QWidget):
    def __init__(self, tr_func):
        super().__init__()
        self.tr = tr_func
        self.coco_images = []
        self.coco_json_path = None
        self.current_image_index = -1

        self.thumbnail_layout = QHBoxLayout(self)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)

        self.prev_thumbnail_label = QLabel()
        self.prev_thumbnail_label.setFixedSize(100, 100)
        self.prev_thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev_thumbnail_label.setFrameShape(QFrame.Shape.StyledPanel)

        self.current_thumbnail_label = QLabel()
        self.current_thumbnail_label.setFixedSize(120, 120)
        self.current_thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_thumbnail_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.current_thumbnail_label.setStyleSheet("border: 2px solid blue;")

        self.next_thumbnail_label = QLabel()
        self.next_thumbnail_label.setFixedSize(100, 100)
        self.next_thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_thumbnail_label.setFrameShape(QFrame.Shape.StyledPanel)

        self.thumbnail_layout.addWidget(self.prev_thumbnail_label)
        self.thumbnail_layout.addWidget(self.current_thumbnail_label)
        self.thumbnail_layout.addWidget(self.next_thumbnail_label)

    def set_coco_data(self, coco_data, coco_json_path):
        self.coco_data = coco_data
        self.coco_json_path = coco_json_path
        self.coco_images = self.coco_data.get("images", []) if self.coco_data else []

    def update_thumbnails(self, current_index):
        self.current_image_index = current_index
        self._update_thumbnail(self.prev_thumbnail_label, self.current_image_index - 1)
        self._update_thumbnail(self.current_thumbnail_label, self.current_image_index)
        self._update_thumbnail(self.next_thumbnail_label, self.current_image_index + 1)

    def _update_thumbnail(self, label, index):
        if 0 <= index < len(self.coco_images):
            image_info = self.coco_images[index]
            image_filename = image_info["file_name"]
            if self.coco_json_path:
                image_dir = os.path.dirname(self.coco_json_path)
                image_path = os.path.join(image_dir, image_filename)
            else:
                image_path = image_filename

            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                label.setPixmap(
                    pixmap.scaled(
                        label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                label.setText(self.tr("thumbnail_not_found"))
        else:
            label.clear()
            label.setText("")
