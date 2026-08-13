"""Mode controllers for PanAnnotator.

Currently contains AddManualModeController which encapsulates the
"add-manual" drawing behaviour extracted from `pan_annotator.py`.
"""

from typing import TYPE_CHECKING

import numpy as np
from pycocotools import mask as mask_util
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QMessageBox

# Keep the same minimum size constant here to avoid circular imports
MIN_MANUAL_BOX_SIZE = 8

if TYPE_CHECKING:
    # Avoid runtime import cycle while keeping type hints readable
    from panopticon.pan_annotator import PanAnnotator  # pragma: no cover


class AddManualModeController:
    """Controller responsible for "add-manual" mode interactions.

    This small helper keeps drawing state and performs the exact same
    operations PanAnnotator did before: start drawing on mouse press,
    show a temporary bbox while dragging, and create the final COCO
    annotation on mouse release.
    """

    def __init__(self, annotator: "PanAnnotator"):
        self.annotator = annotator
        self.drawing_bbox_start_pos = None

    def start_drawing(self, label_point):
        # Same checks as original implementation
        chosen_name, category_id = self.annotator.class_selector.get_current_selection()
        if category_id is None:
            self.annotator.statusBar().showMessage(self.annotator.tr("status_no_class_selected"), 2000)
            return
        # Store start position in image coordinates as float to avoid precision loss
        self.drawing_bbox_start_pos = QPointF(label_point.x() / self.annotator.zoom_scale, label_point.y() / self.annotator.zoom_scale)

    def update_drawing(self, label_point):
        # Show temporary bbox while dragging
        if not self.drawing_bbox_start_pos:
            return
        # Validate the point is inside the label rect (same behavior as before)
        if not self.annotator.image_label.rect().contains(label_point):
            return
        # Convert start position back to zoomed coordinates for display
        zoomed_start = QPointF(self.drawing_bbox_start_pos.x() * self.annotator.zoom_scale, self.drawing_bbox_start_pos.y() * self.annotator.zoom_scale)
        self.annotator._display_image_with_annotations(temp_bbox=(zoomed_start, label_point))

    def finish_drawing(self, end_point_label):
        if not self.drawing_bbox_start_pos:
            return

        # Convert end point to image coordinates
        end_image = QPointF(end_point_label.x() / self.annotator.zoom_scale, end_point_label.y() / self.annotator.zoom_scale)

        x1 = int(min(self.drawing_bbox_start_pos.x(), end_image.x()))
        y1 = int(min(self.drawing_bbox_start_pos.y(), end_image.y()))
        x2 = int(max(self.drawing_bbox_start_pos.x(), end_image.x()))
        y2 = int(max(self.drawing_bbox_start_pos.y(), end_image.y()))

        width = x2 - x1
        height = y2 - y1

        min_size = MIN_MANUAL_BOX_SIZE

        if width >= min_size and height >= min_size:
            chosen_name, category_id = self.annotator.class_selector.get_current_selection()
            if category_id is None:
                self.annotator.statusBar().showMessage(self.annotator.tr("status_no_class_selected"), 2000)
                self.drawing_bbox_start_pos = None
                self.annotator._display_image_with_annotations()
                return

            bbox = [x1, y1, width, height]
            new_annotation_id = self.annotator._get_max_annotation_id() + 1

            new_annotation = {
                "id": new_annotation_id,
                "image_id": self.annotator.current_image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            self.annotator.coco_data.setdefault("annotations", []).append(new_annotation)
            self.annotator.unsaved_changes = True
            self.annotator.statusBar().showMessage(self.annotator.tr("status_manual_annotation_added"), 2000)
        else:
            self.annotator.statusBar().showMessage(self.annotator.tr("status_bbox_too_small"), 2000)

        self.drawing_bbox_start_pos = None
        self.annotator._display_image_with_annotations()

    def cancel(self):
        """Cancel an in-progress manual drawing.

        Clears the start position and forces a redraw. Shows a brief
        status message so the user has feedback that the action was
        cancelled.
        """
        if self.drawing_bbox_start_pos is not None:
            self.drawing_bbox_start_pos = None
            # Refresh the display to remove any temporary box
            self.annotator._display_image_with_annotations()
            try:
                self.annotator.statusBar().showMessage(self.annotator.tr("status_manual_annotation_cancelled"), 1500)
            except Exception:
                # Be conservative: if statusBar or translation isn't available,
                # don't raise an error.
                pass


class AddAIModeController:
    """Controller for the single-point add-AI mode.

    This controller handles AI-assisted annotation by segmenting at a point
    and allowing the user to select from generated masks.
    """

    def __init__(self, annotator: "PanAnnotator"):
        self.annotator = annotator

    def handle_press(self, label_point):
        self._segment_and_annotate(label_point)

    def _segment_and_annotate(self, point):
        if not self.annotator.image or not self.annotator.image_label.pixmap():
            return

        image_x = int(point.x() / self.annotator.zoom_scale)
        image_y = int(point.y() / self.annotator.zoom_scale)
        scaled_point = (image_x, image_y)

        chosen_name, category_id = self.annotator.class_selector.get_current_selection()
        if category_id is None:
            self.annotator.statusBar().showMessage(self.annotator.tr("status_no_class_selected"), 2000)
            return

        seg_w, seg_h = self.annotator.segmentation_inference_resolution_widget.get_resolution()

        try:
            masks = self.annotator.segmenter.segment_points(self.annotator.image, positive_points=[list(scaled_point)], size=(seg_h, seg_w), category_name=chosen_name)
        except Exception as e:
            QMessageBox.critical(self.annotator, self.annotator.tr("error_title"), self.annotator.tr("error_segmentation_failed", error=str(e)))
            return

        if masks is None or len(masks) < 3:
            QMessageBox.warning(self.annotator, self.annotator.tr("warning_title"), self.annotator.tr("warning_no_valid_masks"))
            return

        # Temporarily display the segmentation masks
        self.annotator._display_image_with_annotations(overlay_masks=masks)

        from panopticon.widgets.mask_selection_dialog import MaskSelectionDialog

        chosen_mask_index, ok = MaskSelectionDialog.get_mask_index(self.annotator)

        if ok:
            # User made a selection, use the chosen mask
            chosen_mask = masks[chosen_mask_index]
            mask_np = chosen_mask.squeeze().cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0]

            # pycocotools works with Fortran-contiguous arrays
            mask_fortran = np.asfortranarray(mask_np.astype(np.uint8))
            rle = mask_util.encode(mask_fortran)
            bbox_pycocools = mask_util.toBbox(rle)
            bbox = [int(v) for v in bbox_pycocools] if bbox_pycocools is not None else None

            if bbox:
                categories_list = self.annotator.coco_data.get("categories", [])
                existing_ids = {c.get("id") for c in categories_list}
                # The category_id is already selected from the left pane
                if self.annotator.class_selector._custom_id_mode and category_id not in existing_ids:
                    self.annotator.coco_data["categories"].append(
                        {
                            "id": category_id,
                            "name": str(category_id),
                            "supercategory": "object",
                        }
                    )
                    self.annotator.class_selector.populate_categories(self.annotator.coco_data.get("categories", []))

                # Generate a unique annotation ID
                new_annotation_id = self.annotator._get_max_annotation_id() + 1

                if category_id != -1:
                    new_annotation = {
                        "id": new_annotation_id,
                        "image_id": self.annotator.current_image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 1,
                    }
                    # Type: ignore added because static analyzer may expect a specific annotation class
                    self.annotator.coco_data.setdefault("annotations", []).append(new_annotation)  # type: ignore[arg-type]
                    self.annotator.unsaved_changes = True

        # Redraw without the temporary segmentation overlay
        self.annotator._display_image_with_annotations()


class AddAIMultiPointModeController:
    """Controller for the add-AI multi-point mode.

    This controller uses the existing annotator attributes
    (positive_points / negative_points) and reuses the segmentation
    code that was already in `keyPressEvent` and
    `_display_image_with_annotations` to avoid rewriting behaviour.
    """

    def __init__(self, annotator: "PanAnnotator"):
        self.annotator = annotator
        self.positive_points = []
        self.negative_points = []

    def start_point(self, label_point, button):
        # Convert to image coordinates (same as previous inline code)
        image_point = (int(label_point.x() / self.annotator.zoom_scale), int(label_point.y() / self.annotator.zoom_scale))
        if button == Qt.MouseButton.LeftButton:
            self.positive_points.append(image_point)
        else:
            self.negative_points.append(image_point)
        # Force a redraw which will pick up positive/negative points
        self.annotator._display_image_with_annotations()

    def finish_and_add(self):
        # This copies the logic previously embedded in keyPressEvent
        if not self.positive_points:
            return
        seg_w, seg_h = self.annotator.segmentation_inference_resolution_widget.get_resolution()
        chosen_name, category_id = self.annotator.class_selector.get_current_selection()
        try:
            masks = self.annotator.segmenter.segment_points(self.annotator.image, self.positive_points, self.negative_points, size=(seg_h, seg_w), category_name=chosen_name)
        except Exception as e:
            QMessageBox.critical(self.annotator, self.annotator.tr("error_title"), self.annotator.tr("error_segmentation_failed", error=str(e)))
            return
        if masks is not None and len(masks) > 0:
            mask = masks[0].cpu().numpy().astype(np.uint8)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
                rle = mask_util.encode(np.asfortranarray(mask))
                counts = rle.get("counts")
                if isinstance(counts, bytes):
                    rle["counts"] = counts.decode("ascii")
                chosen_name, category_id = self.annotator.class_selector.get_current_selection()
                if category_id is None:
                    self.annotator.statusBar().showMessage(self.annotator.tr("status_no_class_selected"), 2000)
                    return
                ann_id = self.annotator._get_max_annotation_id() + 1
                ann = {"id": ann_id, "image_id": self.annotator.current_image_id, "category_id": category_id, "bbox": bbox, "segmentation": rle, "area": int(np.sum(mask)), "iscrowd": 0}
                self.annotator.coco_data.setdefault("annotations", []).append(ann)
                self.annotator.unsaved_changes = True
                self.reset_points()
                self.annotator._display_image_with_annotations()

    def cancel(self):
        self.reset_points()
        self.annotator._display_image_with_annotations()

    def reset_points(self):
        self.positive_points = []
        self.negative_points = []

    def get_points(self):
        return self.positive_points, self.negative_points

    def has_points(self):
        return bool(self.positive_points or self.negative_points)


class RemoveAnnotationModeController:
    """Controller for the remove-annotation mode.

    This controller handles removing annotations by clicking on them.
    It reuses the existing annotator methods to preserve behavior.
    """

    def __init__(self, annotator: "PanAnnotator"):
        self.annotator = annotator

    def handle_press(self, label_point):
        if not self.annotator.image or not self.annotator.image_label.pixmap():
            return

        image_x = int(label_point.x() / self.annotator.zoom_scale)
        image_y = int(label_point.y() / self.annotator.zoom_scale)

        annotation_to_remove = None
        for annotation in reversed(self.annotator._get_current_image_annotations()):
            bbox = annotation["bbox"]
            x, y, w, h = bbox
            if x <= image_x < x + w and y <= image_y < y + h:
                annotation_to_remove = annotation
                break  # Remove the top-most annotation

        if annotation_to_remove:
            try:
                self.annotator.coco_data["annotations"].remove(annotation_to_remove)
                self.annotator.unsaved_changes = True
            except ValueError:
                pass  # Already removed / not found
            self.annotator.statusBar().showMessage(self.annotator.tr("status_annotation_removed"), 2000)
            self.annotator._display_image_with_annotations()
        else:
            self.annotator.statusBar().showMessage(self.annotator.tr("status_no_annotation_at_point"), 2000)


class EditBoundingBoxModeController:
    """Controller for the edit bounding box mode.

    This controller handles resizing bounding boxes by dragging their edges.
    It manages the editing state and reuses existing annotator methods.
    """

    def __init__(self, annotator: "PanAnnotator"):
        self.annotator = annotator
        self.editing_annotation = None
        self.editing_handle = None
        self.drag_start_pos = None
        self.original_bbox = None

    def handle_press(self, label_point):
        annotation, handle = self._get_handle_at_point(label_point)
        if annotation and handle:
            self.editing_annotation = annotation
            self.editing_handle = handle
            self.drag_start_pos = label_point
            self.original_bbox = self.editing_annotation["bbox"].copy()

    def handle_move(self, label_point):
        if self.editing_annotation and self.drag_start_pos and self.original_bbox:
            if not self.annotator.image_label.rect().contains(label_point):
                return

            delta_x = (label_point.x() - self.drag_start_pos.x()) / self.annotator.zoom_scale
            delta_y = (label_point.y() - self.drag_start_pos.y()) / self.annotator.zoom_scale

            bbox = self.original_bbox.copy()
            min_size = 5  # Minimum size in image pixels

            if self.editing_handle == "left":
                new_x = self.original_bbox[0] + delta_x
                new_w = self.original_bbox[2] - delta_x
                if new_w >= min_size:
                    bbox[0] = new_x
                    bbox[2] = new_w
            elif self.editing_handle == "right":
                new_w = self.original_bbox[2] + delta_x
                if new_w >= min_size:
                    bbox[2] = new_w
            elif self.editing_handle == "top":
                new_y = self.original_bbox[1] + delta_y
                new_h = self.original_bbox[3] - delta_y
                if new_h >= min_size:
                    bbox[1] = new_y
                    bbox[3] = new_h
            elif self.editing_handle == "bottom":
                new_h = self.original_bbox[3] + delta_y
                if new_h >= min_size:
                    bbox[3] = new_h

            self.editing_annotation["bbox"] = bbox
            self.annotator.unsaved_changes = True
            self.annotator._display_image_with_annotations()
        else:  # Hovering
            if self.annotator.image_label.rect().contains(label_point):
                _, handle = self._get_handle_at_point(label_point)
                if handle in ["top", "bottom"]:
                    self.annotator.setCursor(Qt.CursorShape.SizeVerCursor)
                elif handle in ["left", "right"]:
                    self.annotator.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.annotator.unsetCursor()
            else:
                self.annotator.unsetCursor()

    def handle_release(self):
        if self.editing_annotation:
            bbox = self.editing_annotation["bbox"]
            bbox = [int(coord) for coord in bbox]
            self.editing_annotation["bbox"] = bbox
            self.editing_annotation["area"] = bbox[2] * bbox[3]
            self.annotator.unsaved_changes = True

            self.editing_annotation = None
            self.editing_handle = None
            self.drag_start_pos = None
            self.original_bbox = None
            self.annotator.unsetCursor()
            self.annotator.statusBar().showMessage(self.annotator.tr("status_bbox_edited"), 2000)

    def _get_handle_at_point(self, point):
        handle_margin = 2 * self.annotator.zoom_scale  # pixels on screen, scaled with zoom
        for annotation in reversed(self.annotator._get_current_image_annotations()):
            bbox = annotation["bbox"]
            scaled_x = bbox[0] * self.annotator.zoom_scale
            scaled_y = bbox[1] * self.annotator.zoom_scale
            scaled_w = bbox[2] * self.annotator.zoom_scale
            scaled_h = bbox[3] * self.annotator.zoom_scale

            # Check handles with a margin
            on_left_edge = abs(point.x() - scaled_x) < handle_margin and scaled_y <= point.y() <= scaled_y + scaled_h
            on_right_edge = abs(point.x() - (scaled_x + scaled_w)) < handle_margin and scaled_y <= point.y() <= scaled_y + scaled_h
            on_top_edge = abs(point.y() - scaled_y) < handle_margin and scaled_x <= point.x() <= scaled_x + scaled_w
            on_bottom_edge = abs(point.y() - (scaled_y + scaled_h)) < handle_margin and scaled_x <= point.x() <= scaled_x + scaled_w

            if on_left_edge:
                return annotation, "left"
            if on_right_edge:
                return annotation, "right"
            if on_top_edge:
                return annotation, "top"
            if on_bottom_edge:
                return annotation, "bottom"

        return None, None


class ChangeAnnotationModeController:
    """Controller for the change annotation category mode.

    This controller handles changing the category of annotations by clicking on them.
    It reuses existing annotator methods to preserve behavior.
    """

    def __init__(self, annotator: "PanAnnotator"):
        self.annotator = annotator

    def handle_press(self, label_point):
        if not self.annotator.image or not self.annotator.image_label.pixmap():
            return

        image_x = int(label_point.x() / self.annotator.zoom_scale)
        image_y = int(label_point.y() / self.annotator.zoom_scale)
        # Find which annotation is clicked (top-most first)
        annotation_to_change = None
        for annotation in reversed(self.annotator._get_current_image_annotations()):
            bbox = annotation["bbox"]
            x, y, w, h = bbox
            if x <= image_x <= x + w and y <= image_y <= y + h:
                annotation_to_change = annotation
                break

        if annotation_to_change is None:
            self.annotator.statusBar().showMessage(self.annotator.tr("status_no_annotation_at_point"), 2000)
            return

        selection = self.annotator.class_selector.get_current_selection()
        if not selection or selection[0] is None:
            return

        new_category_name, new_category_id = selection
        annotation_to_change["category_id"] = new_category_id
        self.annotator.unsaved_changes = True
        self.annotator.statusBar().showMessage(self.annotator.tr("status_annotation_changed", new_category_name=new_category_name), 2000)
        self.annotator._display_image_with_annotations()
