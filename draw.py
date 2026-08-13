"""Image drawing utilities for Panopticon.

This module provides functions for drawing annotations such as bounding boxes,
segmentation masks, and labels onto images. It uses libraries like OpenCV,
Pillow, and Supervision.

Functions that draw on images either return a new image with annotations
or modify the input numpy array in-place. This distinction is documented in each
function.
"""

import json
import platform
import os

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from pycocotools import mask as mask_util

from panopticon.config import CATEGORIES_FILE

# Build label translation mapping from categories file
label2el_label: dict[str, str] = {}
try:
    with open(CATEGORIES_FILE) as f:
        _categories_data = json.load(f)
        for _category in _categories_data.get("categories", []):
            _name = _category.get("name")
            _el_name = _category.get("el_name")
            if _name and _el_name:
                label2el_label[_name] = _el_name
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    pass

CANDIDATE_SEGMENTATION_COLORS = ["#00FF00", "#FFFF00", "#FF0000"]
CANDIDATE_SEGMENTATION_OVERLAY_OPACITY = 0.25
CANDIDATE_DETECTION_BBOX_COLOR = "#FF00FF"
SEGMENTATION_COLOR = "#00FFFF"
BBOX_COLOR_RGB = "#5090D0"
TRACK_ID_MODULO = 100

# Font path for labels - select based on OS
if platform.system() == "Windows":
    # Windows: Arial supports Greek and is always available
    LABEL_FONT_PATH = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf")
else:
    # Linux: Noto Sans supports Greek
    LABEL_FONT_PATH = "/usr/share/fonts/noto/NotoSans-Regular.ttf"


def overlay_candidate_segmentations(image: Image.Image, overlay_masks: list) -> Image.Image:
    """Overlays candidate segmentation masks on an image.

    Args:
        image (Image.Image): The input PIL Image.
        overlay_masks (list): A list of segmentation masks (e.g., torch tensors)
            to overlay on the image. The list is expected to be ordered by
            confidence, but the function draws them in reverse order (2, 1, 0).

    Returns:
        Image.Image: A new PIL.Image with the segmentation masks overlaid.
    """
    if image is None or overlay_masks is None:
        return image
    base_np = np.array(image)
    for mask_index in [2, 1, 0]:
        overlay_mask = overlay_masks[mask_index]
        mask_np = overlay_mask.squeeze().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        hex_color = CANDIDATE_SEGMENTATION_COLORS[mask_index]
        r, g, b = (
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
        )
        alpha = CANDIDATE_SEGMENTATION_OVERLAY_OPACITY
        mask_bool = mask_np > 0
        if mask_bool.shape != base_np.shape[:2]:
            continue
        color = np.array([r, g, b], dtype=float)
        base_np[mask_bool] = (1 - alpha) * base_np[mask_bool] + alpha * color
    return Image.fromarray(base_np.astype(np.uint8))


def draw_segmentation_mask(image: np.ndarray, segmentation, color_hex: str = SEGMENTATION_COLOR, alpha: float = 0.5) -> np.ndarray:
    """Overlays a single segmentation mask onto an image.

    This function does not modify the input image; it returns a new numpy array.

    Args:
        image (np.ndarray): The input numpy RGB array.
            segmentation: The segmentation to draw. Supported formats are:
                - RLE dict (with 'counts' and 'size').
                - List of polygons (COCO polygon format).
                - Numpy array mask (HxW) or boolean mask. If a numpy array is provided,
                  it must be 2D (H, W) and match the image dimensions.
        color_hex (str): The hex color string for the mask.
        alpha (float): The overlay opacity, in the range [0, 1].

    Returns:
        np.ndarray: A new RGB numpy array with the mask overlaid.

    Raises:
        ValueError: If the image is not a numpy array or if the segmentation
            format is unsupported.
    """
    if image is None or segmentation is None:
        return image

    # Require numpy array input
    if not isinstance(image, np.ndarray):
        raise ValueError("Unsupported image type. draw_segmentation_mask only accepts numpy arrays (RGB).")
    base_np = image.copy().astype(float)

    # Resolve mask to a 2D numpy array
    mask_np = None
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        # RLE
        mask_np = mask_util.decode(segmentation)
    elif isinstance(segmentation, list):
        # polygon list
        h, w = base_np.shape[:2]
        rles = mask_util.frPyObjects(segmentation, h, w)
        mask_np = mask_util.decode(rles)
        if mask_np is not None and mask_np.ndim == 3:
            # multiple rles -> stack; combine
            mask_np = np.sum(mask_np, axis=2)
    elif isinstance(segmentation, np.ndarray):
        mask_np = segmentation
    else:
        raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")

    if mask_np is None:
        return base_np.astype(np.uint8)

    # Normalize mask to 2D boolean
    if mask_np.ndim == 3:
        # collapse channel dimension
        mask_np = np.sum(mask_np, axis=2)
    mask_bool = mask_np > 0

    # If mask and image sizes mismatch, skip overlay
    if mask_bool.shape != base_np.shape[:2]:
        return base_np.astype(np.uint8)

    # Convert hex color to rgb
    try:
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
    except Exception:
        r, g, b = 0, 255, 255
    color = np.array([r, g, b], dtype=float)

    # Apply overlay
    base_np[mask_bool] = (1 - alpha) * base_np[mask_bool] + alpha * color

    return base_np.astype(np.uint8)


def _get_label_name(name: str, language: str) -> str:
    """Translates a category name to the specified language.

    Args:
        name (str): The category name (e.g., "person").
        language (str): The language code (e.g., "en", "el").

    Returns:
        str: The translated category name, or the original name if not found.
    """
    if language == "en":
        return name
    if language == "el":
        return label2el_label.get(name, name)
    return name


def draw_annotations_on_image(image: Image.Image | np.ndarray, annotations: list, categories_list: list, draw_confidence: bool = True, draw_label: bool = True, segmentation_opacity: float = 0.5, bbox_thickness: int = 2, category_colors: dict[str, str] | None = None, show_track_ids: bool = False, language: str = "en") -> np.ndarray:
    """Draws segmentation masks and bounding boxes on an image.

    This function returns a new annotated image and does not modify the original.

    Args:
        image (Image.Image | np.ndarray): The input image in RGB mode.
        annotations (list): A list of COCO-like annotation dicts for the image.
        categories_list (list): A list of COCO category dicts for label lookup.
        draw_confidence (bool): If True, draws confidence scores on the image.
        draw_label (bool): If True, draws category names on bounding boxes.
        show_track_ids (bool): If True, append tracker id to the label (if present in annotation).
        segmentation_opacity (float): The opacity for segmentation masks.
        bbox_thickness (int): The thickness of bounding box lines.
        category_colors (dict[str, str] | None): An optional dict mapping category names to hex color strings.
        language (str): The language code for label translation (e.g., "en", "el").

    Returns:
        np.ndarray: A new RGB numpy array with annotations drawn.

    Raises:
        ValueError: If the image type is not supported or if a segmentation
            format in an annotation is unsupported.
    """
    assert image is not None

    # Convert base image (PIL) to numpy array (RGB)
    if isinstance(image, Image.Image):
        base_np = np.array(image)
    elif isinstance(image, np.ndarray):
        base_np = image.copy()  # Make a copy to avoid modifying the original
    else:
        raise ValueError("Unsupported image type. Must be PIL.Image.Image or np.ndarray")

    # draw part 1 - segmentation masks
    alpha = segmentation_opacity
    for ann in annotations:
        segmentation = ann.get("segmentation")
        if not segmentation:
            continue

        # Get category color
        cat_id = ann.get("category_id")
        category_name = next((c.get("name") for c in categories_list if c.get("id") == cat_id), str(cat_id))
        if category_colors and category_name in category_colors:
            color_hex = category_colors[category_name]
        else:
            color_hex = SEGMENTATION_COLOR

        # Convert hex color to rgb
        try:
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
        except Exception:
            r, g, b = 0, 255, 255

        mask_np = None
        if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
            # RLE format
            mask_np = mask_util.decode(segmentation)
        elif isinstance(segmentation, list):
            # COCO polygon format
            # Ensure image dimensions are available for mask decoding
            h, w = base_np.shape[:2]
            rles = mask_util.frPyObjects(segmentation, h, w)
            mask_np = mask_util.decode(rles)
            if mask_np.ndim == 3:
                mask_np = np.sum(mask_np, axis=2)
        else:
            raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")

        if mask_np is None:
            continue

        assert mask_np.ndim == 2
        mask_bool = mask_np > 0

        if mask_bool.any():
            assert mask_bool.shape == base_np.shape[:2]
            color = np.array([r, g, b], dtype=float)
            base_np[mask_bool] = (1 - alpha) * base_np[mask_bool] + alpha * color

    # draw part 2 - bounding boxes
    if annotations:
        # Group annotations by category
        category_groups = {}
        for ann in annotations:
            cat_id = ann.get("category_id")
            category_name = next((c.get("name") for c in categories_list if c.get("id") == cat_id), str(cat_id))
            if not isinstance(category_name, str):
                category_name = str(category_name)
            if category_name not in category_groups:
                category_groups[category_name] = []
            category_groups[category_name].append(ann)

        # Draw each category group with its color
        for category_name, anns in category_groups.items():
            if category_colors and category_name in category_colors:
                bbox_color_hex = category_colors[category_name]
            else:
                bbox_color_hex = BBOX_COLOR_RGB

            # Convert RGB to BGR for OpenCV
            bbox_color_bgr = bbox_color_hex[5:7] + bbox_color_hex[3:5] + bbox_color_hex[1:3]

            xywh = np.array([ann.get("bbox", [0, 0, 0, 0]) for ann in anns], dtype=float)
            xyxy = xywh.copy()
            if xyxy.size > 0:
                xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
                xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
                class_ids = [ann.get("category_id", 0) for ann in anns]
                detections = sv.Detections(xyxy=xyxy, class_id=np.array(class_ids))
                labels = []
                for ann in anns:
                    raw_label = ""
                    if draw_label:
                        raw_label = _get_label_name(category_name, language)
                    if draw_confidence:
                        conf = ann.get("bbox_confidence")
                        if isinstance(conf, (int, float)):
                            pct = int(round(float(conf) * 100))
                            conf_str = f"{pct}%"
                            if raw_label:
                                raw_label = f"{raw_label} {conf_str}"
                            else:
                                raw_label = conf_str
                    if show_track_ids:
                        tracker_id = ann.get("tracker_id")
                        if tracker_id is not None:
                            display_id = tracker_id % TRACK_ID_MODULO
                            if raw_label:
                                raw_label = f"{raw_label} #{display_id}"
                            else:
                                raw_label = f"#{display_id}"
                    labels.append(raw_label)
                # Determine text color based on background brightness
                r = int(bbox_color_hex[1:3], 16)
                g = int(bbox_color_hex[3:5], 16)
                b = int(bbox_color_hex[5:7], 16)
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = sv.Color.WHITE if brightness < 128 else sv.Color.BLACK
                box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex(bbox_color_bgr), thickness=bbox_thickness)
                # label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2, text_thickness=1, text_color=text_color, color=sv.Color.from_hex(bbox_color_bgr))
                label_annotator = sv.RichLabelAnnotator(font_path=LABEL_FONT_PATH, font_size=16, text_padding=2, text_color=text_color, color=sv.Color.from_hex(bbox_color_bgr))
                base_np = box_annotator.annotate(base_np, detections=detections)
                if (draw_label or draw_confidence) and any(label for label in labels):
                    base_np = label_annotator.annotate(base_np, detections=detections, labels=labels)

    return base_np


def _draw_dashed_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, dash: int, gap: int) -> None:
    """Draws a dashed line on an image in-place.

    Args:
        img (np.ndarray): The image to draw on. This array is modified.
        x1 (int): The starting x-coordinate of the line.
        y1 (int): The starting y-coordinate of the line.
        x2 (int): The ending x-coordinate of the line.
        y2 (int): The ending y-coordinate of the line.
        color (tuple): The BGR color of the line.
        dash (int): The length of the dash in pixels.
        gap (int): The length of the gap in pixels.
    """
    import math

    length = int(math.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    x = x1
    y = y1
    drawn = 0
    draw = True
    while drawn < length:
        seg_len = dash if draw else gap
        for _ in range(seg_len):
            if 0 <= int(round(y)) < img.shape[0] and 0 <= int(round(x)) < img.shape[1]:
                if draw:
                    img[int(round(y)), int(round(x))] = color
            x += dx
            y += dy
            drawn += 1
            if drawn >= length:
                break
        draw = not draw


def draw_dashed_bbox_with_label(image_np: np.ndarray, x: float, y: float, w: float, h: float, confidence: float, class_label: str) -> None:
    """Draws a dashed bounding box with a label on an image in-place.

    Args:
        image_np (np.ndarray): The image to draw on. This array is modified.
        x (float): The top-left x-coordinate of the bounding box.
        y (float): The top-left y-coordinate of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        confidence (float): The confidence score for the label (0.0 to 1.0).
        class_label (str): The label text to display.
    """
    try:
        r = int(CANDIDATE_DETECTION_BBOX_COLOR[1:3], 16)
        g = int(CANDIDATE_DETECTION_BBOX_COLOR[3:5], 16)
        b = int(CANDIDATE_DETECTION_BBOX_COLOR[5:7], 16)
    except Exception:
        r, g, b = 255, 0, 255
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    dash = 6
    gap = 4
    _draw_dashed_line(image_np, x1, y1, x2, y1, (r, g, b), dash, gap)
    _draw_dashed_line(image_np, x1, y2, x2, y2, (r, g, b), dash, gap)
    _draw_dashed_line(image_np, x1, y1, x1, y2, (r, g, b), dash, gap)
    _draw_dashed_line(image_np, x2, y1, x2, y2, (r, g, b), dash, gap)
    if confidence is not None:
        try:
            pct = int(round(confidence * 100))
            label = f"{class_label} - {pct}%"  # Do not translate this
            cv2.putText(image_np, label, (x1, max(12, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 1, cv2.LINE_AA)
        except Exception:
            pass


def draw_plus(img: np.ndarray, x: int, y: int, color: tuple, size: int = 10) -> None:
    """Draws a plus symbol on an image in-place.

    Args:
        img (np.ndarray): The image to draw on. This array is modified.
        x (int): The horizontal center of the symbol.
        y (int): The vertical center of the symbol.
        color (tuple): The BGR color of the symbol.
        size (int): The size of the symbol in pixels.
    """
    half = size // 2
    cv2.line(img, (x - half, y), (x + half, y), color, 2)
    cv2.line(img, (x, y - half), (x, y + half), color, 2)


def draw_minus(img: np.ndarray, x: int, y: int, color: tuple, size: int = 10) -> None:
    """Draws a minus symbol on an image in-place.

    Args:
        img (np.ndarray): The image to draw on. This array is modified.
        x (int): The horizontal center of the symbol.
        y (int): The vertical center of the symbol.
        color (tuple): The BGR color of the symbol.
        size (int): The size of the symbol in pixels.
    """
    half = size // 2
    cv2.line(img, (x - half, y), (x + half, y), color, 2)


class CategoryTraceAnnotator:
    """Manages multiple TraceAnnotator instances, one per category color.

    This class maintains trace history separately for each category, allowing
    traces to be drawn with colors matching the bounding boxes.
    """

    def __init__(self, trace_length: int = 30):
        """Initialize the CategoryTraceAnnotator.

        Args:
            trace_length (int): The number of past positions to show in traces.
        """
        self.trace_length = trace_length
        self._annotators: dict[str, sv.TraceAnnotator] = {}

    def reset(self):
        """Reset all trace annotators, clearing trace history."""
        self._annotators.clear()

    def annotate(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        category_colors: dict[str, str] | None = None,
        class_id_to_name: dict[int, str] | None = None,
        default_color: str = BBOX_COLOR_RGB,
    ) -> np.ndarray:
        """Draw traces for detections using per-category colors.

        Args:
            image (np.ndarray): The input numpy RGB array.
            detections (sv.Detections): A Supervision Detections object with tracker_id.
            category_colors (dict[str, str] | None): Dict mapping category names to hex colors.
            class_id_to_name (dict[int, str] | None): Dict mapping class IDs to category names.
            default_color (str): Default hex color for categories without a specific color.

        Returns:
            np.ndarray: The annotated image with traces drawn.
        """
        if image is None or detections is None or detections.tracker_id is None:
            return image

        if len(detections) == 0:
            return image

        base_np = image.copy()

        # Group detections by category color
        color_groups: dict[str, list[int]] = {}
        for i, class_id in enumerate(detections.class_id):
            category_name = class_id_to_name.get(int(class_id), str(class_id)) if class_id_to_name else str(class_id)
            if category_colors and category_name in category_colors:
                color_hex = category_colors[category_name]
            else:
                color_hex = default_color

            if color_hex not in color_groups:
                color_groups[color_hex] = []
            color_groups[color_hex].append(i)

        # Draw traces for each color group
        for color_hex, indices in color_groups.items():
            # Get or create annotator for this color
            if color_hex not in self._annotators:
                # Convert RGB hex to BGR for supervision
                color_bgr = color_hex[5:7] + color_hex[3:5] + color_hex[1:3]
                self._annotators[color_hex] = sv.TraceAnnotator(
                    trace_length=self.trace_length,
                    color=sv.Color.from_hex(color_bgr),
                )

            # Create subset detections for this color
            subset_detections = detections[indices]
            base_np = self._annotators[color_hex].annotate(base_np, subset_detections)

        return base_np
