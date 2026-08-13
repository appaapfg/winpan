"""Utility module to add bounding boxes to COCO annotations based on segmentation masks.

This module provides functions to compute bounding boxes from segmentation data (polygons or RLE)
for annotations that either lack bounding boxes or when overwrite_existing_bboxes is True.

Dependencies:
    - pycocotools: For RLE mask handling.
    - numpy: For polygon coordinate processing.
"""

from __future__ import annotations

import json

import numpy as np
from pycocotools import mask as maskUtils

from . import util


def add_bboxes_from_segmentation(
    coco_input: str | dict,
    overwrite_existing_bboxes: bool = False,
    image_ids: list[int] | None = None,
) -> dict:
    """Add bounding boxes to COCO annotations based on segmentation masks.

    Computes bounding boxes from segmentation data for annotations that lack them or when
    overwriting is enabled. Supports both polygon and RLE segmentation formats.

    Args:
        coco_input: Path to COCO JSON file (str) or COCO dataset dict. The dataset should
            contain 'annotations' with segmentation data.
        overwrite_existing_bboxes: If True, recompute bbox even if one exists. If False,
            only add bbox to annotations without one. Defaults to False.
        image_ids: Optional list of image IDs to process. If None, process all images.
            Useful for partial updates on large datasets.

    Returns:
        Updated COCO dataset dict with bboxes added/computed from segmentations. The
        'bbox' field is added or updated in each relevant annotation as [x, y, w, h].

    Raises:
        ValueError: If coco_input is invalid or segmentation data is malformed.
        FileNotFoundError: If coco_input is a path and the file does not exist.

    Examples:
        >>> coco = {"annotations": [{"segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]]}]}
        >>> result = add_bboxes_from_segmentation(coco)
        >>> result["annotations"][0]["bbox"]
        [10.0, 10.0, 10.0, 10.0]

    """
    coco_data = util.load_coco(coco_input).dataset

    annotations = coco_data.get("annotations", [])
    target_image_ids = set(image_ids) if image_ids else None

    for ann in annotations:
        # Skip annotations not in the specified image IDs (if provided)
        if target_image_ids is not None and ann.get("image_id") not in target_image_ids:
            continue

        segmentation = ann.get("segmentation")
        if not segmentation:
            continue

        has_bbox = "bbox" in ann and ann["bbox"] is not None
        # Skip if bbox exists and we're not overwriting
        if not overwrite_existing_bboxes and has_bbox:
            continue

        # Compute bbox from segmentation
        bbox = None
        if isinstance(segmentation, list):
            # Polygons: list of lists, each [x1,y1,x2,y2,...]
            all_x = []
            all_y = []
            for poly in segmentation:
                if isinstance(poly, list) and len(poly) >= 6:  # At least 3 points (6 coords)
                    points = np.array(poly).reshape(-1, 2)
                    all_x.extend(points[:, 0])
                    all_y.extend(points[:, 1])
            if all_x and all_y:
                x_min = min(all_x)
                y_min = min(all_y)
                x_max = max(all_x)
                y_max = max(all_y)
                w = x_max - x_min
                h = y_max - y_min
                bbox = [float(x_min), float(y_min), float(w), float(h)]
        elif isinstance(segmentation, dict) and "counts" in segmentation:
            # RLE: dict with 'size' and 'counts'
            rle = segmentation
            bbox_xywh = maskUtils.toBbox(rle)
            bbox = [float(x) for x in bbox_xywh]

        if bbox is not None:
            ann["bbox"] = bbox

    return coco_data


def main():
    """CLI interface for adding bboxes from segmentation.

    Parses command-line arguments and processes the COCO file accordingly.
    Supports optional filtering by image IDs and overwriting existing bboxes.

    Args (via CLI):
        --input_file: Path to input COCO JSON file (required).
        --output_file: Path to output COCO JSON file (required).
        --overwrite: Flag to overwrite existing bounding boxes.
        --image_ids: Space-separated list of image IDs to process (optional).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Add bounding boxes from segmentation masks to COCO annotations.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input COCO JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output COCO JSON file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bounding boxes")
    parser.add_argument("--image_ids", type=int, nargs="*", default=None, help="Specific image IDs to process")

    args = parser.parse_args()

    updated_coco = add_bboxes_from_segmentation(
        coco_input=args.input_file,
        overwrite_existing_bboxes=args.overwrite,
        image_ids=args.image_ids,
    )

    with open(args.output_file, "w") as f:
        json.dump(updated_coco, f, indent=2)

    print(f"Updated COCO saved to {args.output_file}")


if __name__ == "__main__":
    main()
