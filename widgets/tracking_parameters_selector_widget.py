import os

import yaml
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

try:
    from panopticon.config import SETTINGS_FILE
except ImportError:
    from config import SETTINGS_FILE


class TrackingParametersSelectorWidget(QDialog):
    """Dialog to configure ByteTrack tracking parameters.

    Parameters exposed here match supervision's ByteTrack:
    - track_activation_threshold (float) -> detection confidence threshold for creating tracks
    - lost_track_buffer (int) -> frames to buffer a lost track
    - minimum_matching_threshold (float) -> matching similarity threshold (IoU/score)
    - frame_rate (int) -> video frame rate to scale buffers
    - minimum_consecutive_frames (int) -> minimum frames before a track is 'valid'
    """

    def __init__(self, tr_func, parent=None):
        super().__init__(parent)
        self.tr = tr_func
        self.setWindowTitle(self.tr("tracking_parameters_dialog_title"))
        self.setMinimumWidth(350)

        main_layout = QVBoxLayout(self)

        # Description
        self.description_label = QLabel(self.tr("tracking_parameters_dialog_description"))
        self.description_label.setWordWrap(True)
        main_layout.addWidget(self.description_label)

        # track_activation_threshold
        self.track_activation_label = QLabel(self.tr("track_activation_threshold_label"))
        main_layout.addWidget(self.track_activation_label)
        self.track_activation_spin = QDoubleSpinBox()
        self.track_activation_spin.setRange(0.0, 1.0)
        self.track_activation_spin.setSingleStep(0.05)
        self.track_activation_spin.setDecimals(3)
        main_layout.addWidget(self.track_activation_spin)
        self.track_activation_spin.setToolTip(self.tr("track_activation_threshold_tooltip"))

        # minimum_matching_threshold
        self.minimum_matching_label = QLabel(self.tr("minimum_matching_threshold_label"))
        main_layout.addWidget(self.minimum_matching_label)
        self.minimum_matching_spin = QDoubleSpinBox()
        self.minimum_matching_spin.setRange(0.0, 1.0)
        self.minimum_matching_spin.setSingleStep(0.05)
        self.minimum_matching_spin.setDecimals(3)
        main_layout.addWidget(self.minimum_matching_spin)
        self.minimum_matching_spin.setToolTip(self.tr("minimum_matching_threshold_tooltip"))

        # lost_track_buffer
        self.lost_track_buf_label = QLabel(self.tr("lost_track_buffer_label"))
        main_layout.addWidget(self.lost_track_buf_label)
        self.lost_track_buf_spin = QSpinBox()
        self.lost_track_buf_spin.setRange(0, 10000)
        self.lost_track_buf_spin.setSingleStep(1)
        main_layout.addWidget(self.lost_track_buf_spin)
        self.lost_track_buf_spin.setToolTip(self.tr("lost_track_buffer_tooltip"))

        # frame_rate
        self.frame_rate_label = QLabel(self.tr("frame_rate_label"))
        main_layout.addWidget(self.frame_rate_label)
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setRange(1, 240)
        self.frame_rate_spin.setValue(30)
        self.frame_rate_spin.setSingleStep(1)
        main_layout.addWidget(self.frame_rate_spin)
        self.frame_rate_spin.setToolTip(self.tr("frame_rate_tooltip"))

        # minimum_consecutive_frames
        self.min_consec_label = QLabel(self.tr("minimum_consecutive_frames_label"))
        main_layout.addWidget(self.min_consec_label)
        self.min_consec_spin = QSpinBox()
        self.min_consec_spin.setRange(1, 10)
        self.min_consec_spin.setValue(1)
        self.min_consec_spin.setSingleStep(1)
        main_layout.addWidget(self.min_consec_spin)
        self.min_consec_spin.setToolTip(self.tr("minimum_consecutive_frames_tooltip"))

        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton(self.tr("save_button"))
        self.cancel_button = QPushButton(self.tr("cancel_button"))
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Description already contains full parameter info — no external link

        self.load_settings()

    def showEvent(self, event):
        self._prev_values = {
            "track_activation_threshold": self.get_track_activation_threshold(),
            "minimum_matching_threshold": self.get_minimum_matching_threshold(),
            "lost_track_buffer": self.get_lost_track_buffer(),
            "frame_rate": self.get_frame_rate(),
            "minimum_consecutive_frames": self.get_minimum_consecutive_frames(),
        }
        super().showEvent(event)

    # get_all() removed. Use individual getters instead.

    def get_track_activation_threshold(self):
        return float(self.track_activation_spin.value())

    def get_minimum_matching_threshold(self):
        return float(self.minimum_matching_spin.value())

    def get_lost_track_buffer(self):
        return int(self.lost_track_buf_spin.value())

    def get_frame_rate(self):
        return int(self.frame_rate_spin.value())

    def get_minimum_consecutive_frames(self):
        return int(self.min_consec_spin.value())

    # Settings persistence ---------------------------------------------------
    def load_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    settings = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                print(self.tr("status_error_decoding_yaml_default_tracking"))

        # Defaults align with supervision ByteTrack defaults
        # Read from new key, fall back to old one for backward compatibility
        track_activation = settings.get("track_activation_threshold", 0.25)
        min_match = settings.get("minimum_matching_threshold", 0.8)
        lost_buf = settings.get("lost_track_buffer", 30)
        frame_rate = settings.get("frame_rate", 30)
        min_consec = settings.get("minimum_consecutive_frames", 1)

        # Set values clamped to ranges
        self.track_activation_spin.setValue(float(track_activation))
        self.minimum_matching_spin.setValue(float(min_match))
        self.lost_track_buf_spin.setValue(int(lost_buf))
        self.frame_rate_spin.setValue(int(frame_rate))
        self.min_consec_spin.setValue(int(min_consec))

    def save_settings(self):
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    existing = yaml.safe_load(f) or {}
                    settings = existing
            except yaml.YAMLError:
                print(self.tr("status_error_decoding_yaml_overwriting"))

        settings["track_activation_threshold"] = self.get_track_activation_threshold()
        settings["minimum_matching_threshold"] = self.get_minimum_matching_threshold()
        settings["lost_track_buffer"] = self.get_lost_track_buffer()
        settings["frame_rate"] = self.get_frame_rate()
        settings["minimum_consecutive_frames"] = self.get_minimum_consecutive_frames()

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(settings, f, indent=4)
        except OSError as e:
            print(self.tr("status_error_saving_settings_to_file").format(file=SETTINGS_FILE, e=e))

    def accept(self):
        self.save_settings()
        super().accept()

    def reject(self):
        if hasattr(self, "_prev_values"):
            prev = self._prev_values
            self.track_activation_spin.setValue(prev.get("track_activation_threshold", self.track_activation_spin.value()))
            self.minimum_matching_spin.setValue(prev.get("minimum_matching_threshold", self.minimum_matching_spin.value()))
            self.lost_track_buf_spin.setValue(prev.get("lost_track_buffer", self.lost_track_buf_spin.value()))
            self.frame_rate_spin.setValue(prev.get("frame_rate", self.frame_rate_spin.value()))
            self.min_consec_spin.setValue(prev.get("minimum_consecutive_frames", self.min_consec_spin.value()))
        super().reject()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("tracking_parameters_dialog_title"))
        self.description_label.setText(self.tr("tracking_parameters_dialog_description"))
        self.track_activation_label.setText(self.tr("track_activation_threshold_label"))
        self.minimum_matching_label.setText(self.tr("minimum_matching_threshold_label"))
        self.lost_track_buf_label.setText(self.tr("lost_track_buffer_label"))
        self.frame_rate_label.setText(self.tr("frame_rate_label"))
        self.min_consec_label.setText(self.tr("minimum_consecutive_frames_label"))
        self.save_button.setText(self.tr("save_button"))
        self.cancel_button.setText(self.tr("cancel_button"))
        # Retranslate tooltips
        self.track_activation_spin.setToolTip(self.tr("track_activation_threshold_tooltip"))
        self.minimum_matching_spin.setToolTip(self.tr("minimum_matching_threshold_tooltip"))
        self.lost_track_buf_spin.setToolTip(self.tr("lost_track_buffer_tooltip"))
        self.frame_rate_spin.setToolTip(self.tr("frame_rate_tooltip"))
        self.min_consec_spin.setToolTip(self.tr("minimum_consecutive_frames_tooltip"))
