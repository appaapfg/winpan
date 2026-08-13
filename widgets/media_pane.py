from threading import Lock

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QToolButton, QVBoxLayout, QWidget


class FrameRenderTask(QRunnable):
    def __init__(self, frame, zoom_factor, zoom_center, callback):
        super().__init__()
        self.setAutoDelete(True)
        self._frame = frame
        self._zoom_factor = zoom_factor
        self._zoom_center = zoom_center
        self._callback = callback

    def run(self):
        try:
            frame_array = np.ascontiguousarray(self._frame)
            if frame_array.ndim != 3 or frame_array.shape[2] < 3:
                raise ValueError("Frame must be RGB with 3 channels")

            h, w, ch = frame_array.shape
            frame_size = QSize(w, h)
            bytes_per_line = ch * w
            base_qimage = QImage(frame_array.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            zoom_qimage = None
            zoom_center = self._zoom_center
            zoom_factor = max(1.0, float(self._zoom_factor))

            if zoom_factor > 1.0:
                zoom_w = max(1, int(round(w / zoom_factor)))
                zoom_h = max(1, int(round(h / zoom_factor)))

                cx = zoom_center.x() * w
                cy = zoom_center.y() * h

                half_w = zoom_w / 2.0
                half_h = zoom_h / 2.0

                x0 = int(round(max(0.0, min(cx - half_w, w - zoom_w))))
                y0 = int(round(max(0.0, min(cy - half_h, h - zoom_h))))
                x1 = min(w, x0 + zoom_w)
                y1 = min(h, y0 + zoom_h)

                if x1 <= x0 or y1 <= y0:
                    raise ValueError("Invalid zoom crop bounds computed")

                zoom_array = np.ascontiguousarray(frame_array[y0:y1, x0:x1])
                zoom_h, zoom_w, zoom_ch = zoom_array.shape
                zoom_bytes_per_line = zoom_ch * zoom_w
                zoom_qimage = QImage(
                    zoom_array.data,
                    zoom_w,
                    zoom_h,
                    zoom_bytes_per_line,
                    QImage.Format_RGB888,
                ).copy()

                actual_cx = (x0 + zoom_w / 2.0) / w if w else 0.5
                actual_cy = (y0 + zoom_h / 2.0) / h if h else 0.5
                zoom_center = QPointF(actual_cx, actual_cy)

            self._callback(
                {
                    "frame_size": frame_size,
                    "base_image": base_qimage,
                    "zoom_image": zoom_qimage,
                    "zoom_factor": zoom_factor,
                    "zoom_center": zoom_center,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._callback({"error": exc})


class FrameRenderer(QObject):
    frameProcessed = Signal(object)
    frameFailed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._lock = Lock()
        self._is_busy = False
        self._pending_request = None

    def request_render(self, frame, zoom_factor, zoom_center):
        if frame is None:
            return

        frame_copy = np.ascontiguousarray(frame)
        request = (frame_copy, zoom_factor, zoom_center)

        with self._lock:
            if self._is_busy:
                self._pending_request = request
                return
            self._is_busy = True

        self._start_task(request)

    def _start_task(self, request):
        frame, zoom_factor, zoom_center = request
        task = FrameRenderTask(frame, zoom_factor, zoom_center, self._on_task_finished)
        self._pool.start(task)

    def _on_task_finished(self, result):
        if isinstance(result, dict) and "error" in result:
            self.frameFailed.emit(str(result["error"]))
        else:
            self.frameProcessed.emit(result)

        next_request = None
        with self._lock:
            if self._pending_request is not None:
                next_request = self._pending_request
                self._pending_request = None
            else:
                self._is_busy = False

        if next_request is not None:
            self._start_task(next_request)


class ZoomPreview(QLabel):
    centerChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_size = QSize()
        self._zoom_factor = 1.0
        self._center = QPointF(0.5, 0.5)
        self._is_dragging = False

        self.setCursor(Qt.OpenHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(20, 20, 20, 180);" "border: 1px solid rgba(255, 255, 255, 180);")

    def update_preview(self, base_pixmap, zoom_factor, center, frame_size):
        self._frame_size = frame_size
        self._zoom_factor = zoom_factor
        self._center = center

        if base_pixmap.isNull():
            self.clear()
            return

        preview_pixmap = base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if zoom_factor > 1.0 and frame_size.width() > 0 and frame_size.height() > 0:
            painter = QPainter(preview_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)
            rect = self._calculate_zoom_rect()
            painter.drawRect(rect)
            painter.end()

        self.setPixmap(preview_pixmap)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._is_dragging = True
            self.setCursor(Qt.ClosedHandCursor)
            self._emit_position(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            self._emit_position(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_dragging:
            self._is_dragging = False
            self.setCursor(Qt.OpenHandCursor)
            self._emit_position(event.position())
        super().mouseReleaseEvent(event)

    def _calculate_zoom_rect(self):
        img_w = self._frame_size.width()
        img_h = self._frame_size.height()
        preview_w = self.width()
        preview_h = self.height()

        if img_w == 0 or img_h == 0 or self._zoom_factor <= 1.0:
            return QRectF()

        scale = min(preview_w / img_w, preview_h / img_h)
        scaled_width = img_w * scale
        scaled_height = img_h * scale
        offset_x = (preview_w - scaled_width) / 2.0
        offset_y = (preview_h - scaled_height) / 2.0

        rect_width = scaled_width / self._zoom_factor
        rect_height = scaled_height / self._zoom_factor

        center_x = offset_x + self._center.x() * scaled_width
        center_y = offset_y + self._center.y() * scaled_height

        rect_x = center_x - rect_width / 2.0
        rect_y = center_y - rect_height / 2.0

        rect_x = max(offset_x, min(rect_x, offset_x + scaled_width - rect_width))
        rect_y = max(offset_y, min(rect_y, offset_y + scaled_height - rect_height))

        return QRectF(rect_x, rect_y, rect_width, rect_height)

    def _emit_position(self, pos):
        if self._frame_size.width() <= 0 or self._frame_size.height() <= 0:
            return

        preview_w = self.width()
        preview_h = self.height()
        img_w = self._frame_size.width()
        img_h = self._frame_size.height()

        scale = min(preview_w / img_w, preview_h / img_h)
        scaled_width = img_w * scale
        scaled_height = img_h * scale
        offset_x = (preview_w - scaled_width) / 2.0
        offset_y = (preview_h - scaled_height) / 2.0

        rel_x = (pos.x() - offset_x) / scaled_width if scaled_width else 0.5
        rel_y = (pos.y() - offset_y) / scaled_height if scaled_height else 0.5

        rel_x = max(0.0, min(rel_x, 1.0))
        rel_y = max(0.0, min(rel_y, 1.0))

        self.centerChanged.emit(rel_x, rel_y)


class MediaDisplayWidget(QWidget):
    fastBackwardClicked = Signal()
    fastForwardClicked = Signal()
    def __init__(self, tr_function, parent=None):
        super().__init__(parent)
        self.tr = tr_function
        self._zoom_factor = 1.0
        self._zoom_center = QPointF(0.5, 0.5)
        self._frame_size = QSize()
        self._latest_frame = None
        self._latest_full_image = None
        self._latest_zoom_image = None

        self.media_label = QLabel()
        self.media_label.setAlignment(Qt.AlignCenter)
        self.media_label.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.media_label)
        self.setLayout(layout)

        self.zoom_button = QToolButton(self.media_label)
        self.zoom_button.setText("1x")
        self.zoom_button.setCheckable(True)
        self.zoom_button.setChecked(False)
        self.zoom_button.clicked.connect(self._toggle_zoom_mode)
        self.zoom_button.setStyleSheet("QToolButton {" "background-color: rgba(30, 30, 30, 180);" "color: white;" "padding: 4px 10px;" "border-radius: 4px;" "}" "QToolButton:checked {" "background-color: rgba(220, 80, 60, 200);" "}")

        self.fast_backward_button = QToolButton(self.media_label)
        self.fast_backward_button.setText("<<")
        self.fast_backward_button.clicked.connect(self._on_fast_backward)
        self.fast_backward_button.setStyleSheet("QToolButton {" "background-color: rgba(30, 30, 30, 180);" "color: white;" "padding: 4px 10px;" "border-radius: 4px;" "}")
        self.fast_backward_button.hide()

        self.fast_forward_button = QToolButton(self.media_label)
        self.fast_forward_button.setText(">>")
        self.fast_forward_button.clicked.connect(self._on_fast_forward)
        self.fast_forward_button.setStyleSheet("QToolButton {" "background-color: rgba(30, 30, 30, 180);" "color: white;" "padding: 4px 10px;" "border-radius: 4px;" "}")
        self.fast_forward_button.hide()

        self.zoom_preview = ZoomPreview(self.media_label)
        self.zoom_preview.resize(160, 120)
        self.zoom_preview.hide()
        self.zoom_preview.centerChanged.connect(self._set_zoom_center)

        self.zoom_button.raise_()
        self.fast_backward_button.raise_()
        self.fast_forward_button.raise_()
        self.zoom_preview.raise_()

        self._frame_renderer = FrameRenderer(self)
        self._frame_renderer.frameProcessed.connect(self._on_frame_processed)
        self._frame_renderer.frameFailed.connect(self._on_frame_failed)

    def _on_fast_backward(self):
        self.fastBackwardClicked.emit()

    def _on_fast_forward(self):
        self.fastForwardClicked.emit()

    def show_video_controls(self, show):
        if show:
            self.fast_backward_button.show()
            self.fast_forward_button.show()
        else:
            self.fast_backward_button.hide()
            self.fast_forward_button.hide()

    def set_display_text_key(self, text_key, **kwargs):
        self.media_label.setText(self.tr(text_key, **kwargs))

    def start_source(self):
        self.set_display_text_key("media_capturing")

    def stop_source(self):
        self.set_display_text_key("media_disconnected")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_overlay_geometry()
        self._update_display_pixmap()

    def eventFilter(self, watched, event):
        if watched is self.media_label and event.type() == QEvent.Resize:
            self._update_overlay_geometry()
            self._update_display_pixmap()
        return super().eventFilter(watched, event)

    def _update_overlay_geometry(self):
        label_width = max(0, self.media_label.width())
        label_height = max(0, self.media_label.height())

        button_size = self.zoom_button.sizeHint()
        margin = 12
        button_x = margin
        button_y = max(margin, label_height - button_size.height() - margin)
        self.zoom_button.move(button_x, button_y)

        # Position video control buttons if visible
        current_x = button_x + button_size.width() + margin
        if self.fast_backward_button.isVisible():
            backward_size = self.fast_backward_button.sizeHint()
            self.fast_backward_button.move(current_x, button_y)
            current_x += backward_size.width() + margin
        if self.fast_forward_button.isVisible():
            forward_size = self.fast_forward_button.sizeHint()
            self.fast_forward_button.move(current_x, button_y)

        if label_width < 80 or label_height < 80:
            self.zoom_preview.hide()
            return

        max_preview_width = max(80, label_width // 4)
        max_preview_height = max(60, label_height // 4)

        preview_width = min(180, max_preview_width)
        preview_height = min(140, max_preview_height)

        preview_width = min(preview_width, label_width - (2 * margin))
        preview_height = min(preview_height, label_height - (2 * margin))

        if self._frame_size.width() > 0 and self._frame_size.height() > 0:
            aspect_ratio = self._frame_size.height() / self._frame_size.width()
            if aspect_ratio > 0:
                adjusted_height = preview_width * aspect_ratio
                if adjusted_height > preview_height:
                    adjusted_height = preview_height
                    preview_width = adjusted_height / aspect_ratio if aspect_ratio else preview_width
                else:
                    preview_height = adjusted_height

        preview_width = max(40, int(round(preview_width)))
        preview_height = max(40, int(round(preview_height)))

        if preview_width <= 0 or preview_height <= 0:
            self.zoom_preview.hide()
            return

        self.zoom_preview.resize(preview_width, preview_height)
        self.zoom_preview.move(margin, margin)
        self._refresh_preview()

    def display_processed_frame(self, frame_rgb):
        """Receives a processed RGB frame and schedules it for display."""
        if frame_rgb is None:
            self.set_display_text_key("media_display_error")
            return

        frame_copy = np.ascontiguousarray(frame_rgb)
        self._latest_frame = frame_copy
        self._frame_renderer.request_render(frame_copy, self._zoom_factor, self._zoom_center)

    def _toggle_zoom_mode(self):
        new_factor = 1.0 if self._zoom_factor > 1.0 else 2.0
        self._set_zoom_factor(new_factor)

    def _set_zoom_factor(self, factor):
        self._zoom_factor = 1.0 if factor <= 1.0 else 2.0
        self.zoom_button.setChecked(self._zoom_factor > 1.0)
        self.zoom_button.setText(f"{int(self._zoom_factor)}x")

        if self._zoom_factor <= 1.0:
            self.zoom_preview.hide()

        self._request_render_latest()
        self._update_overlay_geometry()

    def _set_zoom_center(self, norm_x, norm_y):
        self._zoom_center = QPointF(norm_x, norm_y)
        if self._zoom_factor > 1.0:
            self._request_render_latest()

    def _request_render_latest(self):
        if self._latest_frame is None:
            return
        self._frame_renderer.request_render(self._latest_frame, self._zoom_factor, self._zoom_center)

    def _on_frame_processed(self, payload):
        base_image = payload.get("base_image")
        if base_image is None:
            return

        self._frame_size = payload.get("frame_size", self._frame_size)
        self._latest_full_image = base_image
        self._latest_zoom_image = payload.get("zoom_image")
        processed_center = payload.get("zoom_center")
        if isinstance(processed_center, QPointF):
            self._zoom_center = processed_center

        self._update_overlay_geometry()
        self._update_display_pixmap()

    def _on_frame_failed(self, error_message):
        print(f"Frame render error: {error_message}")
        self.set_display_text_key("media_display_error")
        self.zoom_preview.hide()

    def _update_display_pixmap(self):
        target_image = None
        if self._zoom_factor > 1.0 and self._latest_zoom_image is not None:
            target_image = self._latest_zoom_image
        elif self._latest_full_image is not None:
            target_image = self._latest_full_image

        if target_image is None:
            return

        pixmap = QPixmap.fromImage(target_image)
        label_size = self.media_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            self.media_label.setPixmap(pixmap)
            return

        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.media_label.setPixmap(scaled)
        self.zoom_button.raise_()
        self.fast_backward_button.raise_()
        self.fast_forward_button.raise_()
        self.zoom_preview.raise_()

    def _refresh_preview(self):
        if self._zoom_factor <= 1.0 or self._latest_full_image is None:
            self.zoom_preview.hide()
            return

        base_pixmap = QPixmap.fromImage(self._latest_full_image)
        self.zoom_preview.update_preview(
            base_pixmap,
            self._zoom_factor,
            self._zoom_center,
            self._frame_size,
        )
        self.zoom_preview.show()
        self.zoom_preview.raise_()
