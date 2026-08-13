"""Generate segmentation masks for COCO annotations missing a segmentation field.

This module provides a single high-level function:

    generate_missing_segmentations(
        coco_input: str | COCO | dict,
        images_root: str,
        output_json_path: str | None = None,
        model_type: str = "sam2_base",
        batch_size: int = 4,
        size: int = 1024,
        device: str | None = None,
        overwrite_existing: bool = False,
        mask_type: str = "RLE",
    ) -> COCO

Behavior:
1. Accepts either a path to a COCO JSON file or a pycocotools COCO object.
2. Finds all annotations that either lack a "segmentation" key, have it empty, or (if overwrite_existing=True) regardless.
3. Groups those pending annotations by image_id and loads the corresponding images.
4. For each image, runs the SAM Segmenter (see my_inference.segment.Segmenter) to obtain a mask per bounding box.
5. Stores the mask as COCO RLE (compressed) in the annotation's "segmentation" field and (re)computes the "area" from the mask (pixel count).
6. Writes the updated dataset to output_json_path if provided (and not None / "none").

Notes / Limitations:
- Produces RLE segmentations (not polygons) for reliability. Downstream COCO tools accept both.
- If mask generation fails or yields an empty mask, a rectangular mask matching the bounding box is synthesized.
- Bounding boxes in COCO are [x, y, w, h] in absolute pixel coordinates; these are converted to xyxy for the segmenter.
- The Segmenter currently performs per-image segmentation (no multi-image batching). We still batch *annotations* by image.

CLI usage example:
    python data_engine/segment_coco.py \
        --coco_json path/to/annotations.coco.json \
        --output_json path/to/with_segmentations.coco.json \
        --model_type sam2_base --batch_size 4 --size 1024

Examples:
    # Programmatic usage
    from panopticon.data_engine.segment_coco import generate_missing_segmentations
    updated_coco = generate_missing_segmentations(
        coco_input='path/to/annotations.coco.json',
        images_root='path/to/images/',
        output_json_path='path/to/output.coco.json',
        model_type='sam2_base'
    )
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm

from panopticon.data_engine.util import load_coco
from panopticon.my_inference.segment import Segmenter


@dataclass
class PendingAnnotation:
    ann_index: int  # index within coco.dataset["annotations"] list (for in-place update)
    image_id: int
    bbox_xywh: list[float]


def _load_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None


def _bbox_xywh_to_xyxy(b: Iterable[float]) -> list[float]:
    """Convert a bounding box from xywh format to xyxy format.

    Args:
        b (Iterable[float]): The bounding box in [x, y, w, h] format.

    Returns:
        list[float]: The bounding box in [x1, y1, x2, y2] format.
    """
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a boolean / 0-1 mask (H,W) to COCO RLE dict (counts as ascii).

    Args:
        mask (np.ndarray): The binary mask array.

    Returns:
        dict: The COCO RLE dictionary with 'counts' as ascii string.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # pycocotools expects Fortran contiguous
    rle = maskUtils.encode(np.asfortranarray(mask))
    # rle['counts'] is bytes; ensure ascii str for JSON
    rle["counts"] = rle["counts"].decode("ascii")  # type: ignore[index]
    return rle


def generate_missing_segmentations(
    coco_input: str | COCO | dict,
    images_root: str,
    output_json_path: str | None = None,
    model_type: str = "sam2_base",
    batch_size: int = 4,
    size: int = 1024,
    device: str | None = None,
    overwrite_existing: bool = False,
    mask_type: str = "RLE",
) -> dict:
    """Fill in segmentation masks for annotations missing them.

    Args:
        coco_input: Path to input COCO JSON file or a pycocotools COCO object or coco dict.
        images_root: Optional path to the root directory containing images. If provided, overrides automatic detection.
        output_json_path: Where to write updated JSON. If None / "none", file is not written.
        model_type: Segmenter model key (see my_inference.segment.MODEL_CONFIGS).
        batch_size: Number of images to keep loaded & processed per internal loop (currently per-image segmentation anyway).
        size: Inference resolution supplied to segmenter.
        device: Optional device override ("cuda", "cpu"). Auto if None.
        overwrite_existing: If True, regenerate segmentation even if already present.

    Returns:
        The (possibly) modified COCO dataset dict.
    """
    coco = load_coco(coco_input)
    image_root = images_root

    # Collect annotations needing segmentation
    pending: list[PendingAnnotation] = []
    anns = coco.dataset.get("annotations", [])
    for idx, ann in enumerate(anns):
        has_seg = "segmentation" in ann and ann["segmentation"] not in (None, [], {})
        if has_seg and not overwrite_existing:
            continue
        # Require bbox
        if "bbox" not in ann:
            continue
        pending.append(PendingAnnotation(ann_index=idx, image_id=ann["image_id"], bbox_xywh=ann["bbox"]))

    if not pending:
        if output_json_path and output_json_path.lower() not in ("none",):
            # Still optionally write out identical file
            with open(output_json_path, "w") as f:
                json.dump(coco.dataset, f)
        return coco.dataset

    # Group pending by image_id
    by_image: dict[int, list[PendingAnnotation]] = {}
    for p in pending:
        by_image.setdefault(p.image_id, []).append(p)

    # Prepare segmenter
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = Segmenter(model_type=model_type, device=device)

    image_ids = list(by_image.keys())

    # Process images (simple batching over list slice, though segmentation itself is per-image)
    for start in tqdm(range(0, len(image_ids), batch_size), desc="Generating segmentations"):
        batch_ids = image_ids[start : start + batch_size]
        for img_id in batch_ids:
            meta_list = coco.loadImgs([img_id])
            if not meta_list:
                continue
            meta = meta_list[0]
            img_path = os.path.join(image_root, meta["file_name"]) if image_root else meta["file_name"]
            img = _load_image(img_path)
            if img is None:
                continue
            pend_for_img = by_image[img_id]
            # Build tensor of boxes (convert to xyxy)
            xyxy_boxes = torch.tensor([_bbox_xywh_to_xyxy(p.bbox_xywh) for p in pend_for_img], dtype=torch.float32, device=segmenter.model.device if hasattr(segmenter.model, "device") else None)
            if xyxy_boxes.numel() == 0:
                continue
            # Get category labels for each annotation (for SAM3 Class Aware)
            category_labels = []
            for p in pend_for_img:
                ann = anns[p.ann_index]
                cat_id = anns[p.ann_index].get("category_id")
                category_labels.append(coco.cats[cat_id]["name"])
            try:
                masks = segmenter.segment_boxes(img, xyxy_boxes, size=size, category_labels=category_labels)  # (N,H,W)
            except Exception as e:  # noqa: BLE001
                print(e)
                assert False
                # Fall back: create rectangular masks from bboxes
                masks = []
                w_img, h_img = img.size
                for box in xyxy_boxes.tolist():
                    x1, y1, x2, y2 = (int(round(v)) for v in box)
                    x1 = max(0, min(x1, w_img - 1))
                    y1 = max(0, min(y1, h_img - 1))
                    x2 = max(0, min(x2, w_img))
                    y2 = max(0, min(y2, h_img))
                    rect = np.zeros((h_img, w_img), dtype=np.uint8)
                    rect[y1:y2, x1:x2] = 1
                    masks.append(torch.from_numpy(rect))
                masks = torch.stack(masks, dim=0)

            # Ensure tensor on CPU for numpy conversion
            if isinstance(masks, torch.Tensor):
                masks_cpu = masks.detach().cpu()
            else:
                # Unexpected type; skip
                continue

            # Align each mask back to its annotation
            for pending_ann, mask_t in zip(pend_for_img, masks_cpu):
                mask_np = (mask_t > 0).numpy().astype(np.uint8)
                area = int(mask_np.sum())
                # If empty, synthesize rectangular mask
                if area == 0:
                    x1, y1, w, h = pending_ann.bbox_xywh
                    x2 = x1 + w
                    y2 = y1 + h
                    w_img, h_img = img.size
                    rect = np.zeros((h_img, w_img), dtype=np.uint8)
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    x1i = max(0, min(x1i, w_img - 1))
                    y1i = max(0, min(y1i, h_img - 1))
                    x2i = max(0, min(x2i, w_img))
                    y2i = max(0, min(y2i, h_img))
                    rect[y1i:y2i, x1i:x2i] = 1
                    mask_np = rect
                    area = int(mask_np.sum())

                if mask_type.lower() == "rle":
                    segmentation = _mask_to_rle(mask_np)
                    iscrowd = 1
                elif mask_type.lower() == "polygon":
                    polygons = sv.mask_to_polygons(mask_np)
                    # COCO expects a list of lists for polygons
                    segmentation = [p.flatten().tolist() for p in polygons]
                    iscrowd = 0  # Polygons typically imply iscrowd=0
                else:
                    raise ValueError(f"Unsupported mask_type: {mask_type}. Must be 'RLE' or 'polygon'.")

                anns[pending_ann.ann_index]["segmentation"] = segmentation
                anns[pending_ann.ann_index]["area"] = float(area)
                anns[pending_ann.ann_index]["iscrowd"] = iscrowd

    # Write out if requested
    if output_json_path and output_json_path.lower() not in ("none",):
        with open(output_json_path, "w") as f:
            json.dump(coco.dataset, f)

    return coco.dataset


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate segmentations for COCO annotations missing them.")
    p.add_argument("--input_file", type=str, required=True, help="Path to input COCO annotation JSON")
    p.add_argument("--output_file", type=str, default="none", help="Where to save updated JSON (default: none)")
    p.add_argument("--model_type", type=str, default="sam2_tiny", help="Segmenter model type (e.g. sam2_base, sam_base)")
    p.add_argument("--batch_size", type=int, default=4, help="Image batch iteration size")
    p.add_argument("--size", type=int, default=1024, help="Segmentation inference size")
    p.add_argument("--device", type=str, default=None, help="Inference device override")
    p.add_argument("--overwrite_existing", action="store_true", help="Recompute even if segmentation exists")
    p.add_argument("--mask_type", type=str, default="RLE", help="Type of mask to generate: RLE or polygon (default: RLE)")
    return p


def main():  # pragma: no cover - CLI entry
    parser = _build_arg_parser()
    args = parser.parse_args()
    generate_missing_segmentations(
        coco_input=args.input_file,
        images_root=os.path.dirname(args.input_file),
        output_json_path=args.output_file,
        model_type=args.model_type,
        batch_size=args.batch_size,
        size=args.size,
        device=args.device,
        overwrite_existing=args.overwrite_existing,
        mask_type=args.mask_type,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
