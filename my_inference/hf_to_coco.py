from __future__ import annotations
print('==============10==============', flush=True)
from collections.abc import Callable, Iterable
print('==============11==============', flush=True)
import numpy as np
import supervision as sv
import torch
from pycocotools import mask as _coco_mask
print('==============12==============', flush=True)

def _to_python_int_list(arr: Iterable) -> list[int]:
    out: list[int] = []
    for v in arr:
        if isinstance(v, torch.Tensor):
            out.append(int(v.item()))
        else:
            out.append(int(v))
    return out


def _to_python_float_list(arr: Iterable) -> list[float]:
    out: list[float] = []
    for v in arr:
        if isinstance(v, torch.Tensor):
            out.append(float(v.item()))
        else:
            out.append(float(v))
    return out


def _boxes_xyxy_to_xywh(boxes_xyxy: Iterable) -> list[list[float]]:
    xywh_list: list[list[float]] = []
    for box in boxes_xyxy:
        if isinstance(box, torch.Tensor):
            x1, y1, x2, y2 = (float(b.item()) for b in box)
        else:
            x1, y1, x2, y2 = (float(b) for b in (box.tolist() if hasattr(box, "tolist") else box))
        xywh_list.append([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)])
    return xywh_list


def _binary_mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    """Convert 2D binary mask to uncompressed COCO RLE {counts: [...], size: [H, W]} (fallback).

    Notes:
      - Expects a boolean or 0/1 array. Any non-zero is treated as 1.
      - Encodes in row-major order starting with the count of 0s.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    h, w = mask.shape
    arr = (mask.astype(np.uint8) > 0).astype(np.uint8).ravel(order="C")
    counts: list[int] = []
    run_val = 0  # start with zeros
    run_len = 0
    for v in arr:
        if v == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = int(v)
            run_len = 1
    counts.append(run_len)
    return {"counts": counts, "size": [int(h), int(w)]}


def _binary_mask_to_coco_rle_and_area(mask: np.ndarray) -> tuple[dict, float]:
    """Convert a 2D binary mask to COCO RLE using pycocotools when available.

    Returns a tuple: (rle_dict_json_safe, area_float)
    - rle_dict_json_safe has 'counts' as a UTF-8 string (not bytes) and 'size' [H, W].
    - area is computed from the RLE (number of on-pixels).
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    h, w = mask.shape
    bin_mask = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if _coco_mask is not None:
        # pycocotools expects Fortran-contiguous array
        rle_bytes = _coco_mask.encode(np.asfortranarray(bin_mask))
        # rle_bytes['counts'] is bytes; keep a JSON-safe copy for output
        counts_bytes = rle_bytes.get("counts")
        counts_str = counts_bytes.decode("ascii") if isinstance(counts_bytes, (bytes, bytearray)) else str(counts_bytes)
        area = float(_coco_mask.area(rle_bytes))
        rle_json = {"counts": counts_str, "size": [int(h), int(w)]}
        return rle_json, area
    # Fallback to uncompressed RLE
    rle = _binary_mask_to_uncompressed_rle(bin_mask)
    area = float(_area_from_rle_counts(rle["counts"]))
    return rle, area


def _area_from_rle_counts(counts: Iterable[int]) -> int:
    """Compute area (# of 1s) from uncompressed RLE counts list."""
    # odd indices correspond to runs of 1s when starting with zeros
    total = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            total += int(c)
    return total


def HFdetections2COCOdetections(
    detections: dict | list[dict] | None,
    *,
    id_to_name: Callable[[int], str] | None = None,
    start_annotation_id: int = 1,
    include_area: bool = True,
    mask_type: str = "RLE",
) -> tuple[list[dict], list[dict]]:
    """Convert model detections (HF/Ultralytics/RFDETR-style) to COCO-like annotations and categories.

    Inputs (single image expected):
      - detections: dict with keys 'boxes' (xyxy), 'labels', 'scores', and optionally 'tracker_id'.
        Can also be a list and the first item will be used.
      - id_to_name: optional function mapping category_id -> name. Falls back to config.id2label.
      - start_annotation_id: starting integer for 'id' field in output annotations.
      - include_area: whether to compute and include COCO 'area' for each bbox.
      - mask_type: "RLE", "polygon", or "ignore". Determines how masks are encoded in segmentation. Defaults to "RLE".

    Returns:
      (annotations, categories)
        - annotations: list of dicts with keys: id, category_id, bbox [x, y, w, h], bbox_confidence,
          (optional) tracker_id, (optional) segmentation, area
        - categories: list of dicts with keys: id, name, supercategory

    Notes:
      - This function is non-destructive and handles empty inputs gracefully.
      - Boxes are expected in absolute xyxy format; they will be converted to COCO xywh.
      - If 'tracker_id' is present in detections, it will be included in each annotation.
    """

    if detections is None:
        return [], []

    # If a list is provided, pick the first element (single image assumption)
    if isinstance(detections, list):
        if not detections:
            return [], []
        det = detections[0]
    else:
        det = detections

    if not isinstance(det, dict):
        return [], []

    boxes = det.get("boxes")
    labels = det.get("labels")
    scores = det.get("scores")
    masks = det.get("masks")  # Optional, could be list/np/tensor with shape (N, H, W)

    # Normalize to tensors for consistent downstream handling
    if boxes is None or labels is None or scores is None:
        return [], []

    # Prepare Python-native lists
    # boxes stays as iterable for conversion function
    label_list = _to_python_int_list(labels if isinstance(labels, Iterable) else [])
    score_list = _to_python_float_list(scores if isinstance(scores, Iterable) else [])
    bbox_xywh_list = _boxes_xyxy_to_xywh(boxes if isinstance(boxes, Iterable) else [])
    # Normalize masks into a list of numpy arrays (if provided)
    mask_list: list[np.ndarray] | None = None
    if masks is not None:
        mask_list = []
        if isinstance(masks, torch.Tensor):
            # Expect (N, H, W) or (H, W); convert to numpy bool
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            for m in masks:
                mask_list.append(m.detach().cpu().numpy().astype(np.uint8) > 0)
        elif isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = masks[None, ...]
            for i in range(masks.shape[0]):
                mask_list.append(masks[i].astype(np.uint8) > 0)
        elif isinstance(masks, Iterable):
            for m in masks:
                if isinstance(m, torch.Tensor):
                    mask_list.append(m.detach().cpu().numpy().astype(np.uint8) > 0)
                elif isinstance(m, np.ndarray):
                    mask_list.append(m.astype(np.uint8) > 0)
                else:
                    try:
                        arr = np.array(m)
                        mask_list.append(arr.astype(np.uint8) > 0)
                    except Exception:
                        # skip invalid mask
                        continue

    n = min(len(label_list), len(score_list), len(bbox_xywh_list))
    if mask_list is not None:
        n = min(n, len(mask_list))
    if n == 0:
        return [], []

    # Build categories, preferring provided text_labels (if present); no global mapping fallback
    unique_label_ids = sorted(set(label_list[:n]))

    # Attempt to map category_id -> text label name from detections
    name_map: dict[int, str] = {}
    text_labels = det.get("text_labels")
    if isinstance(text_labels, Iterable):
        try:
            tl_list = []
            for v in text_labels:
                if isinstance(v, torch.Tensor):
                    # If a scalar tensor sneaks in, convert to python str
                    tl_list.append(str(v.item()))
                else:
                    tl_list.append(str(v))
            m = min(n, len(tl_list))
            for i in range(m):
                cid = int(label_list[i])
                name = tl_list[i]
                if name and cid not in name_map:
                    name_map[cid] = name
        except Exception:
            # If anything goes wrong, silently fall back to id_to_name
            name_map = {}

    categories = [
        {
            "id": cid,
            "name": name_map.get(cid, (id_to_name(cid) if id_to_name else f"<Unknown ID: {cid}>")),
            "supercategory": "object",
        }
        for cid in unique_label_ids
    ]

    # Extract tracker_id if available
    tracker_ids = det.get("tracker_id")
    tracker_id_list: list[int | None] | None = None
    if tracker_ids is not None:
        tracker_id_list = []
        if isinstance(tracker_ids, np.ndarray):
            tracker_id_list = [int(tid) if tid is not None else None for tid in tracker_ids]
        elif isinstance(tracker_ids, Iterable):
            for tid in tracker_ids:
                if tid is None:
                    tracker_id_list.append(None)
                elif isinstance(tid, torch.Tensor):
                    tracker_id_list.append(int(tid.item()))
                else:
                    tracker_id_list.append(int(tid))

    # Build annotations
    annotations: list[dict] = []
    for i in range(n):
        x, y, w, h = bbox_xywh_list[i]
        ann = {
            "id": start_annotation_id + i,
            "category_id": int(label_list[i]),
            "bbox": [x, y, w, h],
            "bbox_confidence": float(score_list[i]),
        }
        if tracker_id_list is not None and i < len(tracker_id_list) and tracker_id_list[i] is not None:
            ann["tracker_id"] = tracker_id_list[i]
        # Optional segmentation
        if mask_list is not None and i < len(mask_list) and mask_type != "ignore":
            mask = mask_list[i]
            if mask_type == "RLE":
                try:
                    rle, area = _binary_mask_to_coco_rle_and_area(mask)
                    ann["segmentation"] = rle
                    if include_area:
                        ann["area"] = float(area)
                except Exception:
                    pass
            elif mask_type == "polygon":
                try:
                    area = float(np.sum(mask > 0))
                    polygons = sv.mask_to_polygons(mask)
                    ann["segmentation"] = [poly.flatten().tolist() for poly in polygons]
                    if include_area:
                        ann["area"] = area
                except Exception:
                    pass
        annotations.append(ann)

    return annotations, categories
