# Note to LLMs: use QT,
# do not include comments about changelog, such as added, removed, changed, moved, modified
# do not comment code while removing it, just remove it
"""Main entry point for the Panopticon object detection application.

This module initializes the main window, handles the application lifecycle,
and coordinates video streaming, object detection, and UI updates.
"""
import json
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import supervision as sv

# import torch
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from panopticon.config import CATEGORIES_FILE, VERSION, SETTINGS_FILE
import yaml
from panopticon.draw import CategoryTraceAnnotator, draw_annotations_on_image
from panopticon.favorites_utils import (
    FavoritesDialog,
    load_and_apply_favorite_settings,
)
from panopticon.image_streamer import ImageStreamer, StreamToggleButton
from panopticon.my_inference import HFdetections2COCOdetections
from panopticon.my_inference.detect import Detector
from panopticon.translations_data import TRANSLATIONS
from panopticon.video_stream_handler import (  # noqa
    VideoStreamHandler,
    VideoStreamHandlerDummy,
)
from panopticon.widgets.about_panopticon_dialog import AboutDialog
from panopticon.widgets.bbox_thickness_selector_widget import (
    BBoxThicknessSelectorWidget,
)
from panopticon.widgets.confidence_threshold_selector_widget import (
    ConfidenceThresholdSelectorWidget,
)
from panopticon.widgets.detection_filters_widget import (
    FILTER_CATEGORIES,
    DetectionFiltersWidget,
)
from panopticon.widgets.device_selector import DeviceSelector
from panopticon.widgets.fps_config_dialog import (
    FPSConfigDialog,
    calculate_fps_interval,
    get_configured_fps,
)
from panopticon.widgets.inference_resolution_selector_widget import (
    InferenceResolutionSelectorWidget,
)
from panopticon.widgets.language_selector import LanguageActionGroup
from panopticon.widgets.media_pane import MediaDisplayWidget
from panopticon.widgets.model_selector import ModelSelector
from panopticon.widgets.persistent_checkable_action import PersistentCheckableAction
from panopticon.widgets.screenshot_settings_dialog import (
    ScreenshotSettingsDialog,
    get_auto_screenshot_cooldown,
    get_selected_screenshot_path,
)
from panopticon.widgets.segmentation_opacity_selector_widget import (
    SegmentationOpacitySelectorWidget,
)
from panopticon.widgets.tracking_parameters_selector_widget import TrackingParametersSelectorWidget
from panopticon.widgets.theme_manager import ThemeActionGroup
from panopticon.widgets.url_selector_groupbox import URLSelectorGroupBox

try:
    from plyer import notification as plyer_notification
except ImportError:
    plyer_notification = None

FILTER_ANNOATIONS_BASED_ON_TRACKER = True


class DetectionWorker(QObject):
    """Worker class for running object detection in a separate thread.

    Attributes:
        detectionFinished (Signal): Emitted when detection completes successfully.
            Payload is a dictionary containing detections and the original request.
        detectionFailed (Signal): Emitted when detection fails. Payload is the exception.
    """

    detectionFinished = Signal(object)
    detectionFailed = Signal(object)

    def process(self, request):
        """Processes a detection request.

        Args:
            request (dict): A dictionary containing:
                - detector: The detector instance to use.
                - frame: The image frame to process.
                - detect_args: Arguments for the detector.
                - threshold: Confidence threshold.
                - resolution: Inference resolution.
                - nms_iou: NMS IOU threshold.

        Raises:
            RuntimeError: If detector is not provided.
        """
        detector = request.get("detector")
        if detector is None:
            self.detectionFailed.emit(RuntimeError("Detector not provided"))
            return

        try:
            detections = detector.detect(
                request["frame"],
                request["detect_args"],
                threshold=request["threshold"],
                size=request["resolution"],
                nms_iou=request["nms_iou"],
            )
            self.detectionFinished.emit({"detections": detections, "request": request})
        except Exception as exc:
            self.detectionFailed.emit(exc)


NOTIFICATION_COOLDOWN = 2


class PanopticonMainWindow(QMainWindow):
    """Main window for the Panopticon application.

    This class manages the UI, video streaming, and interaction with the
    object detection worker.
    """

    detectionRequested = Signal(object)

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance
        self.ui_elements = {}
        self.current_language = "en"
        self.last_raw_frame = None
        self.last_annotated_frame = None
        self.last_detections = None
        self.last_auto_screenshot_time = None
        self.last_notification_time = None

        self.resize(1280, 720)

        # --- add status bar -- -
        self.status_bar = self.statusBar()
        self.fps_label = QLabel("FPS: --/--")
        self.status_bar.addPermanentWidget(self.fps_label)
        self.frame_timestamps = deque(maxlen=20)  # Store timestamps of the last 20 frames
        # ----------------------

        self.video_stream_handler = VideoStreamHandler()
        # self.video_stream_handler = VideoStreamHandlerDummy()
        self.image_streamer = ImageStreamer()

        self._detection_in_progress = False
        self._pending_rgb_frame = None
        self.detection_worker = DetectionWorker()
        self.detector_thread = QThread(self)
        self.detection_worker.moveToThread(self.detector_thread)
        self.detection_worker.detectionFinished.connect(self._on_detection_finished)
        self.detection_worker.detectionFailed.connect(self._on_detection_failed)
        self.detectionRequested.connect(self.detection_worker.process)
        self.detector_thread.start()

        # --- add menu bar ---
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("")
        self.ui_elements["file_menu"] = file_menu
        exit_action = file_menu.addAction("")
        self.ui_elements["exit_action"] = exit_action
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.app_instance.quit)

        configure_menu = menu_bar.addMenu("")
        self.ui_elements["configure_menu"] = configure_menu
        # Top-level quick actions
        favorites_urls_action = configure_menu.addAction("")
        self.ui_elements["favorites_urls_action"] = favorites_urls_action
        configure_menu.addSeparator()

        # Group: Inference/Settings
        inference_menu = configure_menu.addMenu("")
        self.ui_elements["inference_menu"] = inference_menu
        fps_config_action = inference_menu.addAction("")
        self.ui_elements["fps_config_action"] = fps_config_action
        inference_resolution_action = inference_menu.addAction("")
        self.ui_elements["inference_resolution_action"] = inference_resolution_action
        confidence_threshold_action = inference_menu.addAction("")
        self.ui_elements["confidence_threshold_action"] = confidence_threshold_action
        segmentation_opacity_action = inference_menu.addAction("")
        self.ui_elements["segmentation_opacity_action"] = segmentation_opacity_action
        bbox_thickness_action = inference_menu.addAction("")
        self.ui_elements["bbox_thickness_action"] = bbox_thickness_action
        screenshot_settings_action = inference_menu.addAction("")
        self.ui_elements["screenshot_settings_action"] = screenshot_settings_action
        configure_menu.addSeparator()

        # Group: View options
        view_menu = configure_menu.addMenu("")
        self.ui_elements["view_menu"] = view_menu
        self.show_confidence_action = PersistentCheckableAction("", self, "show_confidence", True)
        view_menu.addAction(self.show_confidence_action)
        self.ui_elements["show_confidence_action"] = self.show_confidence_action
        self.show_confidence_action.setShortcut("Ctrl+1")

        self.show_labels_action = PersistentCheckableAction("", self, "show_labels", True)
        view_menu.addAction(self.show_labels_action)
        self.ui_elements["show_labels_action"] = self.show_labels_action
        self.show_labels_action.setShortcut("Ctrl+2")

        # Draw traces is a viewing option — add it immediately after labels
        self.draw_tracking_traces_action = PersistentCheckableAction("", self, "draw_tracking_traces", True)
        view_menu.addAction(self.draw_tracking_traces_action)
        self.draw_tracking_traces_action.setShortcut("Ctrl+3")
        self.ui_elements["draw_tracking_traces_action"] = self.draw_tracking_traces_action

        self.show_track_ids_action = PersistentCheckableAction("", self, "show_track_ids", False)
        view_menu.addAction(self.show_track_ids_action)
        self.ui_elements["show_track_ids_action"] = self.show_track_ids_action
        self.show_track_ids_action.setShortcut("Ctrl+4")

        self.show_right_pane_action = PersistentCheckableAction("", self, "show_right_pane", True)
        view_menu.addAction(self.show_right_pane_action)
        self.ui_elements["show_right_pane_action"] = self.show_right_pane_action
        self.show_right_pane_action.toggled.connect(self._toggle_right_pane)
        self.show_right_pane_action.setShortcut("Ctrl+9")
        configure_menu.addSeparator()

        # Group: Tracking options
        tracking_menu = configure_menu.addMenu("")
        self.ui_elements["tracking_menu"] = tracking_menu
        self.enable_tracking_action = PersistentCheckableAction("", self, "enable_tracking", False)
        self.enable_tracking_action.toggled.connect(self._on_tracking_toggled)
        tracking_menu.addAction(self.enable_tracking_action)
        self.ui_elements["enable_tracking_action"] = self.enable_tracking_action
        # Add a menu action to configure tracking parameters (ByteTrack)
        tracking_parameters_action = tracking_menu.addAction("")
        self.ui_elements["tracking_parameters_action"] = tracking_parameters_action
        tracking_parameters_action.triggered.connect(self._open_tracking_parameters_dialog)

        # Draw tracking traces action has been moved to the View submenu
        configure_menu.addSeparator()

        # --- End shortcuts ---

        # --- Add Detection Model Submenu ---
        detection_model_menu = configure_menu.addMenu("")
        self.ui_elements["detection_model_menu"] = detection_model_menu

        # Clear any saved last-used detection model on startup so no model is
        # auto-selected. This prevents attempts to load a model automatically
        # at application launch.
        try:
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE) as f:
                        settings_data = yaml.safe_load(f) or {}
                except Exception:
                    settings_data = {}

                if "detection_model" in settings_data:
                    settings_data.pop("detection_model", None)
                    try:
                        with open(SETTINGS_FILE, "w") as f:
                            yaml.dump(settings_data, f, default_flow_style=False, sort_keys=False)
                    except Exception as e:
                        print(f"Error writing settings file while clearing detection model: {e}")
        except Exception as e:
            print(f"Error clearing last-used model setting: {e}")

        self.model_selector = ModelSelector(self, self.tr, self.status_bar)
        detection_model_menu.addActions(self.model_selector.get_actions())
        self.model_selector.triggered.connect(self._on_model_switched)
        # --- End Detection Model Submenu ---

        configure_menu.addSeparator()
        # --- Add Color Theme Submenu ---
        self.theme_action_group = ThemeActionGroup(
            configure_menu,  # Parent menu
            self,  # Main window
            self.app_instance,
            self.tr,
            self.status_bar,
        )
        # --- End Color Theme Submenu ---

        # --- Add Language Selector ---
        self.language_action_group = LanguageActionGroup(
            self,
            self.tr,
            self._set_language,
            self.status_bar,
        )
        configure_menu.addMenu(self.language_action_group)

        # --- Add Device Selector ---
        self.device_selector = DeviceSelector(self, self.tr, self.status_bar, settings_key="pan_opticon_device")
        self.device_selector.device_changed.connect(self._on_device_switched)
        configure_menu.addMenu(self.device_selector)
        # --- End Device Selector ---

        # --- Add Help Menu ---
        help_menu = menu_bar.addMenu("")
        self.ui_elements["help_menu"] = help_menu
        about_qt_action = help_menu.addAction("")
        self.ui_elements["about_qt_action"] = about_qt_action
        about_panopticon_action = help_menu.addAction("")
        self.ui_elements["about_panopticon_action"] = about_panopticon_action
        # --- End Help Menu ---

        # --- set up central widget and main layout ---
        self.central_widget = QWidget()
        main_layout = QHBoxLayout(self.central_widget)

        # --- Media Pane ---
        self.media_display_widget = MediaDisplayWidget(self.tr)
        self.ui_elements["media_display_widget"] = self.media_display_widget
        main_layout.addWidget(self.media_display_widget, 3)

        # Connect video control signals
        self.media_display_widget.fastBackwardClicked.connect(self._on_fast_backward)
        self.media_display_widget.fastForwardClicked.connect(self._on_fast_forward)

        # --- Frame processing timer ---
        self.frame_processing_timer = QTimer()
        # --------------------------------

        # --- button pane ---
        button_pane_widget = QWidget()
        self.button_pane_widget = button_pane_widget
        button_layout = QVBoxLayout(button_pane_widget)
        self._toggle_right_pane()

        # --- URL Selector GroupBox ---
        self.url_selector_groupbox = URLSelectorGroupBox(self.tr, self.statusBar(), self.ui_elements)
        self.frame_processing_timer.timeout.connect(self._acquire_process_and_display_frame)
        self.url_selector_groupbox.connect_button.clicked.connect(self._on_connect_button_clicked)
        button_layout.addWidget(self.url_selector_groupbox)
        # -------------------------

        # --- Detection Filters ---
        self.detection_filters_widget = DetectionFiltersWidget(self.tr, self.status_bar, self.central_widget)
        button_layout.addWidget(self.detection_filters_widget)

        # --- Stream Toggle Button ---
        self.stream_button = StreamToggleButton(self.image_streamer, self.tr, self.status_bar.showMessage, self)
        self.ui_elements["stream_button"] = self.stream_button
        button_layout.addWidget(self.stream_button)

        # --- Take Screenshot Button ---
        self.screenshot_button = QPushButton()
        self.ui_elements["screenshot_button"] = self.screenshot_button
        self.screenshot_button.clicked.connect(self._on_take_screenshot_button_clicked)
        button_layout.addWidget(self.screenshot_button)
        # ----------------------------

        # --- Screenshot Shortcut ---
        self.screenshot_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.screenshot_shortcut.activated.connect(self._on_take_screenshot_button_clicked)
        # ---------------------------

        button_layout.addStretch()
        button_pane_widget.setLayout(button_layout)
        main_layout.addWidget(button_pane_widget, 1)
        self.setCentralWidget(self.central_widget)

        favorites_urls_action.triggered.connect(self._open_favorites_dialog)
        fps_config_action.triggered.connect(self._open_fps_config_dialog)
        screenshot_settings_action.triggered.connect(self._open_screenshot_settings_dialog)
        confidence_threshold_action.triggered.connect(self._open_confidence_threshold_dialog)
        inference_resolution_action.triggered.connect(self._open_inference_resolution_dialog)
        segmentation_opacity_action.triggered.connect(self._open_segmentation_opacity_dialog)
        bbox_thickness_action.triggered.connect(self._open_bbox_thickness_dialog)
        about_qt_action.triggered.connect(self._show_about_qt)
        about_panopticon_action.triggered.connect(self._show_about_panopticon_dialog)

        self.confidence_threshold_widget = ConfidenceThresholdSelectorWidget(self.tr)
        self.inference_resolution_widget = InferenceResolutionSelectorWidget(self.tr)
        self.segmentation_opacity_selector_widget = SegmentationOpacitySelectorWidget(self.tr, self)
        self.bbox_thickness_selector_widget = BBoxThicknessSelectorWidget(self.tr, self)
        self.tracking_parameters_widget = TrackingParametersSelectorWidget(self.tr, self)

        self.detector = Detector(
            self.model_selector.get_selected_model(),
            device=self.device_selector.get_selected_device(),
            all_class_names=FILTER_CATEGORIES,
            resolution=self.inference_resolution_widget.get_resolution(),
        )
        # Update detection filter UI to reflect classes supported by this model
        try:
            supported = self.detector.get_supported_class_names() if self.detector is not None else None
            self.detection_filters_widget.update_available_classes(supported)
        except Exception:
            pass

        self.tracker = self._create_tracker_instance() if self.enable_tracking_action.isChecked() else None
        self.trace_annotator = CategoryTraceAnnotator() if self.enable_tracking_action.isChecked() else None

        self.retranslate_ui()

        self._update_fps_indicator()

    def _on_model_switched(self, action):
        """Handle model selection change.

        Args:
            action: The QAction that triggered the change, containing the new model key.
        """
        new_model_key = action.data()
        self.detector = Detector(new_model_key, device=self.device_selector.get_selected_device(), all_class_names=FILTER_CATEGORIES, resolution=self.inference_resolution_widget.get_resolution())
        if self.tracker is not None:
            self.tracker = self._create_tracker_instance()
        if self.trace_annotator is not None:
            self.trace_annotator.reset()
        try:
            supported = self.detector.get_supported_class_names() if self.detector is not None else None
            self.detection_filters_widget.update_available_classes(supported)
        except Exception:
            pass
        if self.status_bar:
            self.status_bar.showMessage(self.tr("status_model_loaded", model_name=new_model_key), 2000)

    def _on_device_switched(self, new_device):
        """Handle device selection change.

        Args:
            new_device (str): The new device to use for detection.
        """
        self.detector = Detector(self.model_selector.get_selected_model(), device=new_device, all_class_names=FILTER_CATEGORIES, resolution=self.inference_resolution_widget.get_resolution())
        if self.tracker is not None:
            self.tracker = self._create_tracker_instance()
        if self.trace_annotator is not None:
            self.trace_annotator.reset()
        try:
            supported = self.detector.get_supported_class_names() if self.detector is not None else None
            self.detection_filters_widget.update_available_classes(supported)
        except Exception:
            pass
        self.statusBar().showMessage(self.tr("status_device_switched", device_name=new_device), 2000)

    def tr(self, key: str, **kwargs) -> str:
        """Translate a key using the current language.

        Args:
            key (str): The translation key.
            **kwargs: Keyword arguments for string formatting.

        Returns:
            str: The translated string.
        """
        # Fallback to English if the key is not found in the current language
        translation = TRANSLATIONS.get(self.current_language, TRANSLATIONS["en"]).get(key)
        if translation is None:  # Key not found in current language, try English
            translation = TRANSLATIONS["en"].get(key, key)  # Fallback to key itself if not in English
        return translation.format(**kwargs) if kwargs else translation

    def _handle_stream_failure(self):
        """Handle stream failure by resetting state and updating UI."""
        self._detection_in_progress = False
        self._pending_rgb_frame = None
        self.frame_processing_timer.stop()
        self.media_display_widget.set_display_text_key("media_capture_error")
        self.url_selector_groupbox.connect_button.setText(self.tr("connect_button"))
        if self.status_bar:
            self.status_bar.showMessage(self.tr("status_stream_error_disconnected"), 3000)
        self.video_stream_handler.disconnect()

    def _on_detection_finished(self, payload):
        """Handle detection completion and update the UI with results.

        Args:
            payload (dict): The detection result payload containing detections and request.
        """
        self._detection_in_progress = False
        detections = None
        if isinstance(payload, dict):
            detections = payload.get("detections")
        else:
            detections = payload
        self.last_detections = detections

        frame_rgb = self._pending_rgb_frame
        self._pending_rgb_frame = None
        if frame_rgb is None:
            frame_rgb = self.last_raw_frame
        if frame_rgb is None:
            return

        try:
            assert detections is not None
            det = detections[0]
            sv_detections = None
            if det is not None:
                # Track
                if self.tracker is not None and "boxes" in det and len(det["boxes"]) > 0:
                    sv_detections_input = sv.Detections(
                        xyxy=det["boxes"].cpu().numpy() if hasattr(det["boxes"], "cpu") else np.array(det["boxes"]),
                        confidence=det["scores"].cpu().numpy() if hasattr(det["scores"], "cpu") else np.array(det["scores"]),
                        class_id=det["labels"].cpu().numpy().astype(int) if hasattr(det["labels"], "cpu") else np.array(det["labels"]).astype(int),
                    )
                    sv_detections = self.tracker.update_with_detections(sv_detections_input)

                    tracker_id_map = {}
                    if sv_detections.tracker_id is not None:
                        for i, tid in enumerate(sv_detections.tracker_id):
                            box_tuple = tuple(sv_detections.xyxy[i].tolist())
                            tracker_id_map[box_tuple] = tid

                    original_boxes = det["boxes"].cpu().numpy() if hasattr(det["boxes"], "cpu") else np.array(det["boxes"])
                    tracker_ids = []
                    for box in original_boxes:
                        box_tuple = tuple(box.tolist())
                        tracker_ids.append(tracker_id_map.get(box_tuple))

                    det = {
                        "boxes": det["boxes"],
                        "scores": det["scores"],
                        "labels": det["labels"],
                        "text_labels": det.get("text_labels", []),
                        "tracker_id": tracker_ids,
                    }
                    if "masks" in detections[0]:
                        det["masks"] = detections[0]["masks"]

                    if FILTER_ANNOATIONS_BASED_ON_TRACKER:
                        mask = [tid is not None for tid in tracker_ids]
                        det = {
                            "boxes": det["boxes"][mask],
                            "scores": det["scores"][mask],
                            "labels": det["labels"][mask],
                            "text_labels": [det["text_labels"][i] for i in range(len(mask)) if mask[i]],
                            "tracker_id": [tracker_ids[i] for i in range(len(mask)) if mask[i]],
                        }
                        if "masks" in det:
                            det["masks"] = det["masks"][mask]

                annotations, categories = HFdetections2COCOdetections(det)
                annotated_np = draw_annotations_on_image(
                    frame_rgb,
                    annotations,
                    categories,
                    draw_confidence=self.show_confidence_action.isChecked(),
                    draw_label=self.show_labels_action.isChecked(),
                    show_track_ids=self.show_track_ids_action.isChecked(),
                    segmentation_opacity=self.segmentation_opacity_selector_widget.get_opacity(),
                    bbox_thickness=self.bbox_thickness_selector_widget.get_thickness(),
                    category_colors=self.detection_filters_widget.get_category_colors(),
                    language=self.current_language,
                )

                if self.trace_annotator is not None and self.tracker is not None and sv_detections is not None and self.draw_tracking_traces_action.isChecked():
                    class_id_to_name = {cat["id"]: cat["name"] for cat in categories}
                    annotated_np = self.trace_annotator.annotate(
                        annotated_np,
                        sv_detections,
                        category_colors=self.detection_filters_widget.get_category_colors(),
                        class_id_to_name=class_id_to_name,
                    )
                assert annotated_np is not None
            else:
                annotated_np = frame_rgb
            self.last_annotated_frame = annotated_np

            detected_label_strings = self.detection_filters_widget.convert_ids_to_labels(detections[0]["labels"])
            if len(detections) > 0:
                try:
                    notifiable_items_found = set(self.detection_filters_widget.get_notifiable()) & set(detected_label_strings)

                    if notifiable_items_found:
                        now = datetime.now()
                        if self.last_notification_time is None or (now - self.last_notification_time).total_seconds() >= NOTIFICATION_COOLDOWN:
                            objects_str = ", ".join(sorted(list(notifiable_items_found)))
                            notification_title = self.tr("notification_title_object_detected")
                            notification_message = self.tr("notification_message_object_detected", objects=objects_str)
                            self._show_os_notification(title=notification_title, message=notification_message)
                            self.last_notification_time = now
                except Exception as e_notify:
                    print(f"Error during notification generation: {e_notify}")

            screenshotable_items_found = set(self.detection_filters_widget.get_screenshotable()) & set(detected_label_strings)
            if screenshotable_items_found:
                now = datetime.now()
                if self.last_auto_screenshot_time is None or (now - self.last_auto_screenshot_time).total_seconds() >= get_auto_screenshot_cooldown() / 1000:
                    self._take_screenshot(
                        self.last_raw_frame,
                        self.last_annotated_frame,
                        self.last_detections,
                        source="automatic",
                    )
                    self.last_auto_screenshot_time = now
        except Exception as e:
            print(f"Error processing detection result in main window: {type(e).__name__}: {e}")
            self._handle_stream_failure()
            return

        try:
            self.media_display_widget.display_processed_frame(self.last_annotated_frame)
            self._update_fps_indicator()
        except Exception as e:
            print(f"Error displaying frame in main window: {type(e).__name__}: {e}")
            self._handle_stream_failure()
            return

        if self.image_streamer.is_running():
            try:
                self.image_streamer.update_frame(self.last_annotated_frame)
            except Exception as e:
                print(f"Error updating frame for streamer: {e}")

    def _on_detection_failed(self, error):
        """Handle detection failure.

        Args:
            error (Exception): The exception that occurred during detection.
        """
        self._detection_in_progress = False
        self._pending_rgb_frame = None
        error_type = type(error).__name__ if isinstance(error, BaseException) else type(error).__name__
        print(f"Error processing frame in detection worker: {error_type}: {error}")
        self._handle_stream_failure()

    def _acquire_process_and_display_frame(self):
        """Acquire, process, and display a frame from the video stream."""
        if not self.video_stream_handler.is_opened():
            self._handle_stream_failure()
            return

        if self._detection_in_progress:
            self.video_stream_handler.grab_frame()
            return

        ret, frame = self.video_stream_handler.read_frame()
        if not ret:
            self._handle_stream_failure()
            return

        if frame is None:
            return

        self.frame_timestamps.append(datetime.now())

        try:
            assert type(frame) is np.ndarray
            np_BGR_frame = np.array(frame)
            assert type(np_BGR_frame) is np.ndarray
            np_RGB_frame = cv2.cvtColor(np_BGR_frame, cv2.COLOR_BGR2RGB)
            self.last_raw_frame = np_RGB_frame.copy()
            self._pending_rgb_frame = np_RGB_frame
        except Exception as e:
            print(f"Error acquiring frame from source: {e}")
            self._handle_stream_failure()
            return

        if not self.model_selector.get_selected_model():
            annotated_np = np_RGB_frame
            self.last_detections = None
            self.last_annotated_frame = annotated_np
            try:
                self.media_display_widget.display_processed_frame(annotated_np)
                self._update_fps_indicator()
            except Exception as e:
                print(f"Error displaying frame in main window: {type(e).__name__}: {e}")
                self._handle_stream_failure()
                return

            if self.image_streamer.is_running():
                try:
                    self.image_streamer.update_frame(annotated_np)
                except Exception as e:
                    print(f"Error updating frame for streamer: {e}")
            self._pending_rgb_frame = None
            return

        detection_request = {
            "frame": np_BGR_frame,
            "detect_args": [
                self.detection_filters_widget.get_detectable(),
            ],
            "threshold": self.confidence_threshold_widget.get_confidence_threshold(),
            "resolution": self.inference_resolution_widget.get_resolution(),
            "nms_iou": self.confidence_threshold_widget.get_nms_iou_threshold(),
            "detector": self.detector,
        }
        self._detection_in_progress = True
        self.detectionRequested.emit(detection_request)

    def _on_take_screenshot_button_clicked(self):
        if self.last_raw_frame is not None and self.last_annotated_frame is not None and self.last_detections is not None:
            self._take_screenshot(self.last_raw_frame, self.last_annotated_frame, self.last_detections, source="manual")
        else:
            self.status_bar.showMessage(self.tr("status_no_frame_to_screenshot"), 3000)

    def _on_connect_button_clicked(self):
        if self.video_stream_handler.is_opened():
            self.frame_processing_timer.stop()
            self.video_stream_handler.disconnect()
            self.media_display_widget.stop_source()
            self.media_display_widget.show_video_controls(False)
            self.url_selector_groupbox.connect_button.setText(self.tr("connect_button"))
            self.status_bar.showMessage(self.tr("status_disconnected"), 3000)
            self.fps_label.setText("FPS: --/--")
            if self.tracker is not None:
                self.tracker.reset()
            if self.trace_annotator is not None:
                self.trace_annotator.reset()
        else:
            camera_url = self.url_selector_groupbox.url_input.text()
            if not camera_url:
                self.status_bar.showMessage(self.tr("error_url_empty"), 3000)
                return

            self.status_bar.showMessage(self.tr("status_connecting", target=camera_url), 3000)
            self.media_display_widget.set_display_text_key("media_connecting")
            QApplication.processEvents()

            if self.video_stream_handler.connect(camera_url):
                self.status_bar.showMessage(
                    self.tr(
                        "status_connected",
                        target=self.video_stream_handler.get_source(),
                    ),
                    3000,
                )
                self.url_selector_groupbox.connect_button.setText(self.tr("disconnect_button"))
                self.media_display_widget.start_source()
                self.media_display_widget.show_video_controls(self.video_stream_handler.is_local_file)

                interval_ms = calculate_fps_interval(self.tr, self.status_bar)
                self.frame_processing_timer.setInterval(interval_ms)
                self.frame_processing_timer.start()
                self._acquire_process_and_display_frame()
                self._update_fps_indicator()

                if self.tracker is not None:
                    self.tracker = self._create_tracker_instance()
                if self.trace_annotator is not None:
                    self.trace_annotator.reset()
            else:
                self.status_bar.showMessage(self.tr("status_connection_failed", target=camera_url), 5000)
                self.media_display_widget.set_display_text_key("media_connection_failed")
                self.url_selector_groupbox.connect_button.setText(self.tr("connect_button"))

    def _update_media_timer_interval(self):
        interval_ms = calculate_fps_interval(self.tr, self.status_bar)
        self.frame_processing_timer.setInterval(interval_ms)
        if self.frame_processing_timer.isActive():
            self.frame_processing_timer.start()
        self._update_fps_indicator()

    def _on_fast_backward(self):
        self.video_stream_handler.seek_by_frames(-240)  # Backward 240 frames

    def _on_fast_forward(self):
        self.video_stream_handler.seek_by_frames(240)  # Forward 240 frames

    def _open_favorites_dialog(self):
        dialog = FavoritesDialog(self, self.tr)
        if dialog.exec():
            load_and_apply_favorite_settings(
                self.url_selector_groupbox.favorites_buttons_widget.get_buttons(),
                self.url_selector_groupbox.url_input,
                self.status_bar,
                self.tr,
            )
            self.status_bar.showMessage(self.tr("status_favorite_urls_updated"), 3000)
        else:
            self.status_bar.showMessage(self.tr("status_favorite_url_config_cancelled"), 3000)

    def _update_fps_indicator(self):
        if len(self.frame_timestamps) > 1:
            time_diff = (self.frame_timestamps[-1] - self.frame_timestamps[0]).total_seconds()
            if time_diff > 0:
                running_fps = (len(self.frame_timestamps) - 1) / time_diff
                configured_fps = get_configured_fps(self.tr, self.status_bar)
                self.fps_label.setText(f"{running_fps:.1f}/{configured_fps:.1f} FPS")
        else:
            configured_fps = get_configured_fps(self.tr, self.status_bar)
            self.fps_label.setText(f"--/{configured_fps:.1f} FPS")

    def _take_screenshot(self, raw_frame, annotated_frame, detections, source=""):
        """Takes a screenshot and saves it to the configured path."""
        try:
            base_screenshot_path = get_selected_screenshot_path()
            if base_screenshot_path:
                os.makedirs(base_screenshot_path, exist_ok=True)
                # Create a subdirectory based on the source
                screenshot_path = os.path.join(base_screenshot_path, source)
                os.makedirs(screenshot_path, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # Save raw image
                raw_filename = f"screenshot_{timestamp}.jpg"
                raw_full_path = os.path.join(screenshot_path, raw_filename)
                cv2.imwrite(raw_full_path, cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))

                # Save annotated image
                annotated_filename = f"screenshot_{timestamp}_annotated.jpg"
                annotated_full_path = os.path.join(screenshot_path, annotated_filename)
                cv2.imwrite(annotated_full_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                # Save annotations as COCO JSON
                self._save_coco_annotations(raw_frame, detections, raw_filename, annotated_filename, timestamp, screenshot_path)

                if self.status_bar:
                    self.status_bar.showMessage(self.tr("status_screenshot_saved", path=annotated_full_path), 3000)
            elif self.status_bar:
                self.status_bar.showMessage(self.tr("status_screenshot_settings_path_not_set_or_invalid"), 3000)
        except Exception as e:
            print(f"Error during screenshot generation: {e}")
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_error_generating_screenshot", e=e), 5000)

    def _save_coco_annotations(self, raw_frame, detections, raw_filename, annotated_filename, timestamp, output_path):
        image_id = 1  # Since we are processing one image at a time
        height, width = raw_frame.shape[:2]

        # Load all known categories from the main COCO file
        try:
            with open(CATEGORIES_FILE) as f:
                panopticon_coco = json.load(f)
            all_categories = panopticon_coco.get("categories", [])
        except (FileNotFoundError, json.JSONDecodeError):
            all_categories = []

        # Create a lookup for existing categories
        name_to_cat = {cat["name"]: cat for cat in all_categories}
        max_id = max([cat["id"] for cat in all_categories] + [-1])

        # Process detected labels and add new categories if necessary
        detected_labels = detections[0]["text_labels"]
        for label_name in set(detected_labels):
            if label_name not in name_to_cat:
                max_id += 1
                new_cat = {"id": max_id, "name": label_name, "supercategory": "other"}
                all_categories.append(new_cat)
                name_to_cat[label_name] = new_cat

        images = [
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": raw_filename,
                "annotated_file_name": annotated_filename,
                "license": 1,
                "date_captured": timestamp,
            }
        ]

        annotations = []
        for i, (box, label_name, score) in enumerate(zip(detections[0]["boxes"], detections[0]["text_labels"], detections[0]["scores"])):
            x1, y1, x2, y2 = box.tolist()
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)
            bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO format: [x, y, width, height]
            area = (x2 - x1) * (y2 - y1)
            category_id = name_to_cat[label_name]["id"]
            annotations.append(
                {
                    "id": i + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "score": score.item(),
                }
            )

        coco_data = {"info": {"description": "Panopticon Detections", "version": "1.0", "year": datetime.now().year, "date_created": datetime.now().isoformat()}, "licenses": [{"id": 1, "name": "Unknown", "url": ""}], "images": images, "annotations": annotations, "categories": all_categories}

        json_filename = f"screenshot_{timestamp}.coco.json"
        json_full_path = os.path.join(output_path, json_filename)
        with open(json_full_path, "w") as f:
            json.dump(coco_data, f, indent=4)

    def _show_os_notification(self, title=None, message=None):
        final_title = title if title is not None else self.tr("default_notification_title")
        final_message = message if message is not None else self.tr("default_notification_message")

        if plyer_notification:
            try:
                plyer_notification.notify(title=final_title, message=final_message, app_name="Panopticon", timeout=5000)
                # Only show status bar message for the generic test notification
                if title is None and message is None:
                    self.status_bar.showMessage(self.tr("status_test_notification_sent"), 3000)
            except Exception as e:
                print(f"Error sending notification with plyer: {e}")
                self.status_bar.showMessage(self.tr("status_error_sending_notification", e=e), 5000)
        else:
            print("Plyer library not found. Cannot send OS notification.")
            print(f"Notification Title: {final_title}")
            print(f"Notification Message: {final_message}")
            # Only show status bar message for the generic test notification
            if title is None and message is None:
                self.status_bar.showMessage(self.tr("status_plyer_not_found_notification"), 5000)

    def _open_fps_config_dialog(self):
        dialog = FPSConfigDialog(self, self.tr)
        if dialog.exec():
            self.status_bar.showMessage(self.tr("status_fps_config_saved_applying"), 3000)
            self._update_media_timer_interval()
        else:
            self.status_bar.showMessage(self.tr("status_fps_config_cancelled"), 3000)

    def _open_screenshot_settings_dialog(self):
        dialog = ScreenshotSettingsDialog(self.tr, self)
        if dialog.exec():
            self.status_bar.showMessage(self.tr("status_screenshot_settings_saved"), 3000)
        else:
            self.status_bar.showMessage(self.tr("status_screenshot_settings_config_cancelled"), 3000)

    def _open_confidence_threshold_dialog(self):
        if self.confidence_threshold_widget.exec():
            self.status_bar.showMessage(self.tr("status_confidence_threshold_saved"), 3000)
            # Recreate tracker with the new tracking threshold immediately so changes take effect
            if self.tracker is not None:
                self.tracker = self._create_tracker_instance()
            if self.trace_annotator is not None:
                self.trace_annotator.reset()
        else:
            self.status_bar.showMessage(self.tr("status_confidence_threshold_config_cancelled"), 3000)

    def _open_inference_resolution_dialog(self):
        if self.inference_resolution_widget.exec():
            self.detector = Detector(self.model_selector.get_selected_model(), device=self.device_selector.get_selected_device(), all_class_names=FILTER_CATEGORIES, resolution=self.inference_resolution_widget.get_resolution())
            try:
                supported = self.detector.get_supported_class_names() if self.detector is not None else None
                self.detection_filters_widget.update_available_classes(supported)
            except Exception:
                pass
            self.status_bar.showMessage(self.tr("status_inference_resolution_saved"), 3000)
        else:
            self.status_bar.showMessage(self.tr("status_inference_resolution_config_cancelled"), 3000)

    def _open_segmentation_opacity_dialog(self):
        self.segmentation_opacity_selector_widget.exec()

    def _open_bbox_thickness_dialog(self):
        self.bbox_thickness_selector_widget.exec()

    def _open_tracking_parameters_dialog(self):
        dialog = self.tracking_parameters_widget
        if dialog.exec():
            self.status_bar.showMessage(self.tr("status_tracking_parameters_saved"), 3000)
            if self.tracker is not None:
                self.tracker = self._create_tracker_instance()
        else:
            self.status_bar.showMessage(self.tr("status_tracking_parameters_config_cancelled"), 3000)

    def _create_tracker_instance(self):
        """Create a sv.ByteTrack instance using the configured tracking parameters.

        Return the constructed ByteTrack object."""
        # Use the activation threshold from the new Tracking Parameters widget
        assert hasattr(self, "tracking_parameters_widget") and self.tracking_parameters_widget is not None
        activation_threshold = self.tracking_parameters_widget.get_track_activation_threshold()
        # Read individual parameters from the tracking widget
        lost_track_buffer = self.tracking_parameters_widget.get_lost_track_buffer()
        minimum_matching_threshold = self.tracking_parameters_widget.get_minimum_matching_threshold()
        frame_rate = self.tracking_parameters_widget.get_frame_rate()
        minimum_consecutive_frames = self.tracking_parameters_widget.get_minimum_consecutive_frames()
        tracker = sv.ByteTrack(
            track_activation_threshold=activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        return tracker

    def _show_about_qt(self):
        QMessageBox.aboutQt(self)

    def _show_about_panopticon_dialog(self):
        dialog = AboutDialog(self.tr, self)
        dialog.exec()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("app_title", version=VERSION))
        self.ui_elements["file_menu"].setTitle(self.tr("file_menu"))
        self.ui_elements["exit_action"].setText(self.tr("exit_action"))
        self.ui_elements["configure_menu"].setTitle(self.tr("configure_menu"))
        self.ui_elements["favorites_urls_action"].setText(self.tr("favorites_urls_action"))
        # Submenu titles
        if "inference_menu" in self.ui_elements:
            self.ui_elements["inference_menu"].setTitle(self.tr("configure_inference_menu"))
        if "view_menu" in self.ui_elements:
            self.ui_elements["view_menu"].setTitle(self.tr("view_menu"))
        if "tracking_menu" in self.ui_elements:
            self.ui_elements["tracking_menu"].setTitle(self.tr("tracking_menu"))
        self.ui_elements["fps_config_action"].setText(self.tr("configure_fps_action"))
        self.ui_elements["screenshot_settings_action"].setText(self.tr("configure_screenshot_settings_action"))
        self.ui_elements["confidence_threshold_action"].setText(self.tr("configure_confidence_threshold_action"))
        self.ui_elements["inference_resolution_action"].setText(self.tr("configure_inference_resolution_action"))
        self.ui_elements["segmentation_opacity_action"].setText(self.tr("segmentation_opacity_action"))
        self.ui_elements["bbox_thickness_action"].setText(self.tr("bbox_thickness_action"))
        self.ui_elements["show_confidence_action"].setText(self.tr("show_confidence_action"))
        self.ui_elements["show_labels_action"].setText(self.tr("draw_labels_action"))
        self.ui_elements["show_track_ids_action"].setText(self.tr("show_track_ids_action"))
        self.ui_elements["show_right_pane_action"].setText(self.tr("show_right_pane_action"))
        self.enable_tracking_action.setText(self.tr("enable_tracking_action"))
        self.draw_tracking_traces_action.setText(self.tr("draw_tracking_traces_action"))
        self.ui_elements["tracking_parameters_action"].setText(self.tr("configure_tracking_parameters_action"))
        self.ui_elements["detection_model_menu"].setTitle(self.tr("select_detection_model_menu"))
        self.language_action_group.retranslate_ui()
        self.device_selector.retranslate_ui()
        self.segmentation_opacity_selector_widget.retranslate_ui()
        self.bbox_thickness_selector_widget.retranslate_ui()
        self.confidence_threshold_widget.retranslate_ui()
        self.tracking_parameters_widget.retranslate_ui()
        # Retranslate theme menu items
        self.theme_action_group.retranslate_ui()

        # Retranslate help menu
        self.ui_elements["help_menu"].setTitle(self.tr("help_menu"))
        self.ui_elements["about_qt_action"].setText(self.tr("about_qt_action"))
        self.ui_elements["about_panopticon_action"].setText(self.tr("about_panopticon_action"))  # Retranslate new action

        # Set initial media label text if not connected
        if not self.video_stream_handler.is_opened():
            self.media_display_widget.set_display_text_key("media_disconnected")
        else:
            if self.media_display_widget.media_label.pixmap() is None:
                self.media_display_widget.set_display_text_key("media_capturing")

        self.url_selector_groupbox.retranslate_ui(self.video_stream_handler.is_opened())
        self.detection_filters_widget.retranslate_ui()
        self.stream_button.retranslate_ui()
        self.ui_elements["screenshot_button"].setText(self.tr("take_screenshot_button"))

        self.status_bar.showMessage(self.tr("status_ready"))

        self.detection_filters_widget.load_settings(self.status_bar)
        self._update_media_timer_interval()
        if self.video_stream_handler.is_opened():
            if self.media_display_widget.media_label.pixmap() is None:
                self.media_display_widget.set_display_text_key("media_capturing")
        else:
            self.media_display_widget.set_display_text_key("media_disconnected")

    def _set_language(self, lang_code):
        self.language_action_group.save_language_setting(lang_code)
        self.retranslate_ui()
        self.language_action_group.set_language(lang_code)

    def closeEvent(self, event):
        if self.video_stream_handler.is_opened():
            self.video_stream_handler.disconnect()
        if self.frame_processing_timer.isActive():
            self.frame_processing_timer.stop()
        if self.detector_thread.isRunning():
            self.detector_thread.quit()
            self.detector_thread.wait()
        event.accept()

    def _toggle_right_pane(self):
        self.button_pane_widget.setVisible(self.show_right_pane_action.isChecked())

    def _on_tracking_toggled(self, enabled: bool):
        """Handle tracking toggle callback.

        Args:
            enabled: Whether tracking is now enabled.
        """
        if enabled:
            self.tracker = self._create_tracker_instance()
            self.trace_annotator = CategoryTraceAnnotator()
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_tracking_enabled"), 2000)
        else:
            self.tracker = None
            self.trace_annotator = None
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_tracking_disabled"), 2000)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("images/pan_opticon_logo_transparent.png"))

    window = PanopticonMainWindow(app)
    app.setStyle("Windows")  # Apply style before showing
    window.show()
    exit_code = app.exec()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
