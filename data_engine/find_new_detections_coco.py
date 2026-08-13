"""Utility to discover NEW detections (not yet annotated) in a COCO dataset.

Description:
    This module provides functionality to identify new object detections in images
    from a COCO-formatted dataset that are not yet annotated. It runs a detection
    model on images, compares predictions against existing ground-truth annotations
    of the same category, and filters out detections that overlap sufficiently with
    existing boxes.

Steps:
    1. Load a COCO annotation JSON (images + existing annotations).
    2. Load images referenced in the COCO file.
    3. Run the chosen detection model (Detector) on each image (batched).
    4. For every predicted box, compare against existing GT boxes of the SAME category in that image.
    5. Keep only predictions whose IoU with any existing annotation of the same category is < iou_threshold.
    6. Return a list of new detections in COCO results format.

Notes:
    - Existing COCO annotation boxes are stored as [x, y, w, h]; they are converted to xyxy for IoU.
    - Detected boxes produced by Detector.detect are already xyxy (float tensors).
    - Category matching is done via category *name* (string). If a detection label is not present in the COCO categories mapping, that detection is skipped (cannot decide duplication).
    - The order of the returned list corresponds to the (filtered) order of loaded images (images that failed to load are skipped).

Example CLI usage:
    python data_engine/find_new_detections_coco.py --coco_json path/to/annotations.coco.json --model_type yoloe_11s --categories all
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm

from panopticon.data_engine.util import load_coco
from panopticon.my_inference.detect import Detector


def _collect_existing_boxes_per_image(coco: COCO):
    """Collect existing annotation boxes per image from a COCO dataset.

    Description:
        Parses the COCO annotations to group existing bounding boxes by image ID, converting them from xywh to xyxy format and associating them with category names.

    Args:
        coco (COCO): Loaded COCO dataset object.

    Returns:
        dict[int, dict]: Dictionary mapping image_id to a dict containing:
            - 'boxes': Tensor[N,4] of xyxy boxes (float32).
            - 'cat_names': List[str] of category names.
            - 'cat_ids': List[int] of category IDs.
    """
    cat_id2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    per_image = {}
    for ann in coco.dataset.get("annotations", []):
        img_id = ann["image_id"]
        bbox_xywh = ann["bbox"]  # [x,y,w,h]
        x, y, w, h = bbox_xywh
        xyxy = [x, y, x + w, y + h]
        if img_id not in per_image:
            per_image[img_id] = {"boxes": [], "cat_names": [], "cat_ids": []}
        per_image[img_id]["boxes"].append(xyxy)
        cat_name = cat_id2name.get(ann["category_id"], None)
        per_image[img_id]["cat_names"].append(cat_name)
        per_image[img_id]["cat_ids"].append(ann["category_id"])

    for img_id, data in per_image.items():
        if data["boxes"]:
            data["boxes"] = torch.tensor(data["boxes"], dtype=torch.float32)
        else:
            data["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
    return per_image


def _prepare_category_name_mapping(coco: COCO):
    """Prepare bidirectional mappings between category names and IDs from COCO dataset.

    Description:
        Extracts category information from the COCO dataset to create mappings
        for converting between category names and IDs.

    Args:
        coco (COCO): Loaded COCO dataset object.

    Returns:
        tuple[dict[str, int], dict[int, str]]: A tuple containing:
            - name2id: Dict mapping category names to IDs.
            - id2name: Dict mapping category IDs to names.
    """
    cats = coco.loadCats(coco.getCatIds())
    name2id = {c["name"]: c["id"] for c in cats}
    id2name = {c["id"]: c["name"] for c in cats}
    return name2id, id2name


def find_new_detections_in_coco(
    coco_input: str | COCO | dict,
    images_root: str,
    model_type: str,
    threshold: float = 0.25,
    batch_size: int = 16,
    inference_resolution: tuple[int, int] = (640, 640),
    iou_threshold: float = 0.5,
    categories: Iterable[str] | None = None,
    device: str | None = None,
    box_prompt_referenced_images: Image.Image | None = None,
    box_prompt_detections: list[dict] | None = None,
    box_prompt_categories: list[dict[str, Any]] | None = None,
    mask_type: str = "RLE",
    save_file: str | None = None,
    round_bbox_coordinates: bool = True,
) -> list[dict[str, Any]]:
    """Run detection over a COCO dataset and return NEW detections in COCO results format.

    Description:
        This function runs object detection on images from a COCO dataset and filters out detections that overlap with existing annotations of the same category (based on IoU threshold).
        It returns only the "new" detections in COCO results JSON format, optionally including segmentation masks.

        A detection is considered *new* if it does NOT have IoU >= iou_threshold with
        ANY existing ground-truth annotation of the SAME category for that image.

    Args:
        coco_input (str | COCO | dict): Path to COCO JSON file, loaded COCO object, or dataset dict.
        images_root (str): Root directory containing the images referenced in the COCO file.
        model_type (str): Model key for the Detector (e.g., 'yoloe_11s').
        threshold (float, optional): Detection confidence threshold. Defaults to 0.25.
        batch_size (int, optional): Number of images per detection batch. Defaults to 16.
        inference_resolution (tuple[int, int], optional): (H, W) for inference. Defaults to (640, 640).
        iou_threshold (float, optional): IoU threshold; detections with IoU >= this are considered already present. Defaults to 0.5.
        categories (Iterable[str] | None, optional): Category names to restrict detection to, or None for all. Defaults to None.
        device (str | None, optional): Device for inference (e.g., 'cpu', 'cuda'). Defaults to None.
        box_prompt_referenced_images (Image.Image | None, optional): Reference images for box prompting. Defaults to None.
        box_prompt_detections (list[dict] | None, optional): Detections for box prompting. Defaults to None.
        box_prompt_categories (list[dict[str, Any]] | None, optional): Categories for box prompting. Defaults to None.
        mask_type (str, optional): Mask format ('RLE' or 'polygon'). Defaults to 'RLE'.
        save_file (str | None, optional): Path to save results JSON. Defaults to None.
        round_bbox_coordinates (bool, optional): Whether to round bounding box coordinates to integers. Defaults to True.

    Returns:
        list[dict[str, Any]]: List of detection dicts in COCO results format containing only new detections.

    Raises:
        ValueError: If provided categories do not exist in the dataset, or if box prompt arguments are incomplete.
    """
    coco = load_coco(coco_input)

    # Generate unique annotation IDs starting from max existing + 1
    existing_annotations = coco.dataset.get("annotations", [])
    max_id = max((ann.get("id", 0) for ann in existing_annotations), default=0)
    next_id = max_id + 1

    name2id, id2name = _prepare_category_name_mapping(coco)

    if categories is None:
        target_categories = sorted(name2id.keys())  # use all COCO categories
    elif categories == "all":  # convenience
        target_categories = sorted(name2id.keys())
    else:
        target_categories = [c for c in categories if c in name2id]
        if not target_categories:
            raise ValueError("None of the provided categories exist in the COCO dataset.")

    image_root = images_root

    all_image_ids = coco.getImgIds()
    if not all_image_ids:
        return []

    existing = _collect_existing_boxes_per_image(coco)

    detector = Detector(model_type=model_type, device=device, batch_size=batch_size, all_class_names=target_categories, resolution=inference_resolution)

    # Handle box prompting
    if box_prompt_referenced_images is not None or box_prompt_detections is not None or box_prompt_categories is not None:
        if not (box_prompt_referenced_images is not None and box_prompt_detections is not None and box_prompt_categories is not None):
            raise ValueError("All three box prompt arguments (box_prompt_referenced_images, box_prompt_detections, box_prompt_categories) must be provided together.")
        detector.set_box_prompt(box_prompt_referenced_images, box_prompt_detections, box_prompt_categories)

    coco_result_detections: list[dict[str, Any]] = []

    # For optional per-image summary (not part of return value)
    per_image_new_counts: dict[int, int] = {}

    for start in tqdm(range(0, len(all_image_ids), batch_size), desc="Detecting"):
        batch_img_ids = all_image_ids[start : start + batch_size]
        images_meta = coco.loadImgs(batch_img_ids)

        batch_imgs = []
        loaded_img_ids_in_batch = []
        for meta in images_meta:
            path = os.path.join(image_root, meta["file_name"]) if image_root else meta["file_name"]
            img = cv2.imread(path)
            if img is None:
                continue
            batch_imgs.append(img)
            loaded_img_ids_in_batch.append(meta["id"])

        if not batch_imgs:
            continue

        # Detector expects list[list[str]] for detection labels (one list per image)
        labels_batch: list[list[str | int]] = [list(target_categories)]  # replicate automatically inside detect if needed
        results = detector.detect(batch_imgs, labels_batch, threshold=threshold, size=inference_resolution)

        for img_idx, det_result in enumerate(results):
            img_id = loaded_img_ids_in_batch[img_idx]
            existing_entry = existing.get(img_id, {"boxes": torch.zeros((0, 4)), "cat_names": []})
            existing_boxes = existing_entry["boxes"]  # (E,4) xyxy
            existing_cat_names = existing_entry.get("cat_names", [])

            # Build detection label names
            if "text_labels" in det_result:
                det_label_names = det_result["text_labels"]
            else:
                # Map numeric labels to strings via detector helper
                det_label_names = detector.label_list_to_str_list(det_result.get("labels", []))

            det_boxes = det_result.get("boxes", torch.zeros((0, 4)))
            det_scores = det_result.get("scores", torch.zeros((0,)))

            if det_boxes.numel() == 0:
                per_image_new_counts[img_id] = 0
                continue

            keep_indices = []
            # Pre-group existing boxes by category for efficiency
            # Build a mask/tensor for each category encountered in detections
            # Ensure existing boxes are on same device as detection boxes to avoid device mismatch errors
            if existing_boxes.device != det_boxes.device:
                existing_boxes = existing_boxes.to(det_boxes.device)
            # Filter detections to keep only those that are "new" (low IoU with existing annotations of same category)
            for idx, (box, score, label_name) in enumerate(zip(det_boxes, det_scores, det_label_names)):
                if label_name not in target_categories:
                    continue  # Not a category we track (or unsupported by dataset)

                # Gather existing boxes for same category
                if existing_boxes.shape[0] == 0:
                    # No existing boxes at all -> keep
                    keep_indices.append(idx)
                    continue

                # Filter existing boxes to same category name
                same_cat_indices = [i for i, cn in enumerate(existing_cat_names) if cn == label_name]
                if not same_cat_indices:
                    keep_indices.append(idx)
                    continue

                same_cat_boxes = existing_boxes[same_cat_indices]
                ious = torchvision.ops.box_iou(box.unsqueeze(0), same_cat_boxes)  # (1, M)
                max_iou = float(ious.max().item()) if same_cat_boxes.shape[0] > 0 else 0.0
                if max_iou < iou_threshold:
                    keep_indices.append(idx)

            new_count_for_image = 0
            if keep_indices:
                for idx in keep_indices:
                    ln = det_label_names[idx]
                    if ln not in name2id:
                        continue
                    cat_id = name2id[ln]
                    box_xyxy = det_boxes[idx]
                    score = float(det_scores[idx].item()) if det_scores.shape[0] > idx else 0.0
                    x1, y1, x2, y2 = (float(v) for v in box_xyxy.tolist())
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if round_bbox_coordinates:
                        x1 = round(x1)
                        y1 = round(y1)
                        w = round(w)
                        h = round(h)
                    det_entry = {
                        "id": next_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, w, h],
                        "bbox_confidence": score,
                        "area": float(w * h),
                        "iscrowd": 1,
                    }
                    # Attach segmentation mask if available
                    if "masks" in det_result and det_result["masks"] is not None:
                        masks_obj = det_result["masks"]
                        # Supported formats: torchvision/ultralytics style tensor (N,H,W) or result.masks.data
                        assert isinstance(masks_obj, torch.Tensor)
                        # Expect shape (N,H,W) or (N,1,H,W)
                        mask_tensor = masks_obj[idx]
                        assert mask_tensor.ndim == 2
                        # Binarize (some models give probabilities)
                        if mask_tensor.dtype != torch.uint8:
                            mask_tensor = (mask_tensor > 0.5).to(torch.uint8)
                        np_mask = mask_tensor.cpu().numpy()

                        if mask_type.lower() == "rle":
                            # Ensure Fortran order for pycocotools
                            rle = maskUtils.encode(np.asfortranarray(np_mask))
                            counts_val = rle.get("counts", None)
                            if isinstance(counts_val, bytes):
                                rle["counts"] = counts_val.decode("ascii")
                            det_entry["segmentation"] = rle
                            det_entry["iscrowd"] = 1
                        elif mask_type.lower() == "polygon":
                            polygons = sv.mask_to_polygons(np_mask)
                            segmentation = [p.flatten().tolist() for p in polygons]
                            det_entry["segmentation"] = segmentation
                            det_entry["iscrowd"] = 0  # Polygons typically imply iscrowd=0
                        else:
                            raise ValueError(f"Unsupported mask_type: {mask_type}. Must be 'RLE' or 'polygon'.")

                    coco_result_detections.append(det_entry)
                    next_id += 1
                    new_count_for_image += 1
            per_image_new_counts[img_id] = new_count_for_image

    if save_file:
        with open(save_file, "w") as f:
            json.dump(coco_result_detections, f, indent=2)

    print(f"finished finding new detections with {model_type}, total new detections: {len(coco_result_detections)}")
    return coco_result_detections


def main():  # CLI wrapper
    parser = argparse.ArgumentParser(description="Find new (unannotated) detections in a COCO dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to COCO annotation JSON")
    parser.add_argument("--model_type", type=str, default="yoloe_11s", help="Model type for Detector")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--inference_resolution", type=int, nargs=2, default=[640, 640], help="Inference resolution H W")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold to consider detection already annotated")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Subset of category names or 'all'")
    parser.add_argument("--mask_type", type=str, default="RLE", help="Type of mask to generate: RLE or polygon (default: RLE)")
    parser.add_argument("--device", type=str, default=None, help="Device for model inference, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--save_file", type=str, default=None, help="Path to save new detections JSON.")
    parser.add_argument("--no_round_bbox_coordinates", action="store_false", dest="round_bbox_coordinates", default=True, help="Do not round bounding box coordinates to integers (default: round)")

    args = parser.parse_args()

    cats = None if args.categories is None else (args.categories if args.categories != ["all"] else "all")

    new_dets = find_new_detections_in_coco(
        coco_input=args.input_file,
        images_root=os.path.dirname(args.input_file),
        model_type=args.model_type,
        threshold=args.threshold,
        batch_size=args.batch_size,
        inference_resolution=tuple(args.inference_resolution),
        iou_threshold=args.iou_threshold,
        categories=cats,
        mask_type=args.mask_type,
        device=args.device,
        save_file=args.save_file,
        round_bbox_coordinates=args.round_bbox_coordinates,
    )
    print(f"Total NEW detections: {len(new_dets)}")
    # Show first 5 detections for quick glance
    print(json.dumps(new_dets[:5], indent=2))


if __name__ == "__main__":
    main()
