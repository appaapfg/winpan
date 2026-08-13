import os

import cv2
import numpy as np
import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from skimage.restoration import estimate_sigma

from panopticon.config import MODEL_DIRECTORY, SETTINGS_FILE
from panopticon.util.avf3 import find_iframe, open_mpv, resolve_video_path


class ImageInfoPane(QWidget):
    def __init__(self, tr_function=None, parent=None):
        super().__init__(parent)
        self.tr = tr_function or (lambda x: x)  # fallback to identity if no tr
        self.current_name = None
        self.current_img_id = None
        self.current_image = None
        self.setFixedWidth(200)
        layout = QVBoxLayout(self)
        self.name_label = QTextEdit()
        self.name_label.setReadOnly(True)
        self.name_label.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.name_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.name_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.name_label.setFrameStyle(QFrame.Shape.NoFrame)
        font = self.name_label.font()
        fm = QFontMetrics(font)
        line_height = fm.lineSpacing()
        self.name_label.setMinimumHeight(3 * line_height)
        self.name_label.setMaximumHeight(3 * line_height)
        self.name_label.mousePressEvent = self._on_name_label_click
        self.width_label = QLabel()
        self.height_label = QLabel()
        self.id_label = QLabel()
        self.entropy_label = QLabel("Image Shannon Entropy: ")
        self.variance_label = QLabel("Variance of the Laplacian: ")
        self.sift_label = QLabel("SIFT Keypoints: ")
        self.fast_label = QLabel("FAST Keypoints: ")
        self.akaze_label = QLabel("AKAZE Keypoints: ")
        self.sigma_label = QLabel("Sigma: ")
        self.brisque_label = QLabel("BRISQUE Score: ")
        layout.addWidget(self.name_label)
        layout.addWidget(self.width_label)
        layout.addWidget(self.height_label)
        layout.addWidget(self.id_label)
        layout.addWidget(self.entropy_label)
        layout.addWidget(self.variance_label)
        layout.addWidget(self.sift_label)
        layout.addWidget(self.fast_label)
        layout.addWidget(self.akaze_label)
        layout.addWidget(self.sigma_label)
        layout.addWidget(self.brisque_label)
        layout.addStretch()

    def _on_name_label_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            text = self.name_label.toPlainText().strip()
            if text.startswith("Name: "):
                name = text[6:]
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._open_video_at_index(name)
                else:
                    QApplication.clipboard().setText(name)

    def _open_video_at_index(self, full_name):
        # remove extension
        name_without_ext = ".".join(full_name.split(".")[:-1])
        parts = name_without_ext.rsplit("_", 1)
        if len(parts) == 2:
            video_name = parts[0]
            index_str = parts[1]
            try:
                target_index = int(index_str) - 1  # 0-based
                # load settings
                settings = {}
                if os.path.exists(SETTINGS_FILE):
                    with open(SETTINGS_FILE) as f:
                        settings = yaml.safe_load(f) or {}
                video_sources = settings.get("video_sources", "")
                if video_sources:
                    candidate_path = os.path.join(video_sources, video_name)
                    resolved_path = resolve_video_path(candidate_path)
                    if resolved_path:
                        raw_ts, rel_ts, _ = find_iframe(resolved_path, target_index)
                        if raw_ts is not None:
                            open_mpv(resolved_path, rel_ts)
            except ValueError:
                pass

    def set_image_info(self, name, img_id, image=None):
        self.current_name = name
        self.current_img_id = img_id
        self.current_image = image
        self.name_label.setPlainText(f"{self.tr('custom_object_id_dialog_name_label')} {name}\n\n")
        self.id_label.setText(f"{self.tr('image_id_label')} {img_id}")
        if image:
            width, height = image.size
            self.width_label.setText(f"{self.tr('inference_resolution_width_label')} {width}")
            self.height_label.setText(f"{self.tr('inference_resolution_height_label')} {height}")
            # self.entropy_label.setText(f"Image Shannon Entropy: {self._calculate_image_entropy(image)}")
            # self.variance_label.setText(f"Variance of the Laplacian: {self._calculate_laplacian_variance(image)}")
            # self.sift_label.setText(f"SIFT Keypoints: {self._calculate_sift_keypoints(image)}")
            # self.fast_label.setText(f"FAST Keypoints: {self._calculate_fast_keypoints(image)}")
            # self.akaze_label.setText(f"AKAZE Keypoints: {self._calculate_akaze_keypoints(image)}")
            # self.sigma_label.setText(f"Sigma: {self._calculate_sigma(image)}")
            # self.brisque_label.setText(f"BRISQUE Score: {self._calculate_brisque_score(image)}")
        else:
            self.width_label.setText(self.tr("inference_resolution_width_label"))
            self.height_label.setText(self.tr("inference_resolution_height_label"))
            # self.entropy_label.setText("Image Shannon Entropy: ")
            # self.variance_label.setText("Variance of the Laplacian: ")
            # self.sift_label.setText("SIFT Keypoints: ")
            # self.fast_label.setText("FAST Keypoints: ")
            # self.akaze_label.setText("AKAZE Keypoints: ")
            # self.sigma_label.setText("Sigma: ")
            # self.brisque_label.setText("BRISQUE Score: ")

    def _calculate_image_entropy(self, image):
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            img_gray = img_array
        entropy = shannon_entropy(img_gray)
        return f"{entropy:.2f}"

    def _calculate_laplacian_variance(self, image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            variance = laplacian.var()
            return f"{variance:.2f}"
        except Exception as e:
            print(f"Error calculating Laplacian variance: {e}")
            return "N/A"

    def _calculate_sift_keypoints(self, image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            sift = cv2.SIFT_create()
            keypoints = sift.detect(img_gray, None)
            return len(keypoints)
        except Exception as e:
            print(f"Error calculating SIFT keypoints: {e}")
            return "N/A"

    def _calculate_fast_keypoints(self, image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            fast = cv2.FastFeatureDetector_create(threshold=30)
            keypoints = fast.detect(img_gray, None)
            return len(keypoints)
        except Exception as e:
            print(f"Error calculating FAST keypoints: {e}")
            return "N/A"

    def _calculate_akaze_keypoints(self, image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            akaze = cv2.AKAZE_create()
            keypoints = akaze.detect(img_gray, None)
            return len(keypoints)
        except Exception as e:
            print(f"Error calculating AKAZE keypoints: {e}")
            return "N/A"

    def _calculate_sigma(self, image):
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_gray = rgb2gray(img_array)
            else:
                img_gray = img_array
            sigma = estimate_sigma(img_gray, channel_axis=None)
            return f"{sigma:.4f}"
        except Exception as e:
            print(f"Error calculating sigma: {e}")
            return "N/A"

    def _calculate_brisque_score(self, image):
        try:
            model_path = os.path.join(MODEL_DIRECTORY, "brisque_model_live.yml")
            range_path = os.path.join(MODEL_DIRECTORY, "brisque_range_live.yml")
            if not os.path.exists(model_path) or not os.path.exists(range_path):
                return "N/A"
            brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            score = brisque.compute(img_array)
            return f"{score[0]:.4f}"
        except Exception as e:
            print(f"Error calculating BRISQUE score: {e}")
            return "N/A"

    def clear(self):
        self.current_name = None
        self.current_img_id = None
        self.current_image = None
        self.name_label.setPlainText(self.tr("custom_object_id_dialog_name_label") + " \n\n")
        self.width_label.setText(self.tr("inference_resolution_width_label"))
        self.height_label.setText(self.tr("inference_resolution_height_label"))
        self.id_label.setText(self.tr("image_id_label"))
        # self.entropy_label.setText("Image Shannon Entropy: ")
        # self.variance_label.setText("Variance of the Laplacian: ")
        # self.sift_label.setText("SIFT Keypoints: ")
        # self.fast_label.setText("FAST Keypoints: ")
        # self.akaze_label.setText("AKAZE Keypoints: ")
        # self.sigma_label.setText("Sigma: ")
        # self.brisque_label.setText("BRISQUE Score: ")

    def retranslate_ui(self):
        if self.current_name is not None:
            self.set_image_info(self.current_name, self.current_img_id, self.current_image)
        else:
            self.clear()
