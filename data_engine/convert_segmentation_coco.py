"""Utilities to convert COCO segmentation encodings between RLE and polygon formats.

This module provides functions to transform segmentation data in COCO annotation files, supporting both RLE (Run-Length Encoding) and polygon formats.

Functions:
    - rle_to_polygons: Convert RLE to polygon coordinates.
    - polygons_to_rle: Convert polygons to RLE.
    - annotation_rle_to_polygon: Mutate annotation to use polygons.
    - annotation_polygon_to_rle: Mutate annotation to use RLE.
    - convert_coco_mask_type: Process entire COCO files.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import supervision as sv
from pycocotools import mask as maskUtils

from panopticon.data_engine.util import load_coco


def rle_to_polygons(rle: dict[str, Any]) -> list[list[float]]:
    """Convert a COCO RLE (dict with 'size' and 'counts') to polygon(s).

    Args:
        rle: COCO-style RLE dict with 'size' (list of [height, width]) and 'counts' (bytes or str). Handles both encoded and decoded counts.

    Returns:
        List of polygons where each polygon is a flat list of floats [x1,y1,x2,y2,...].
    """
    counts = rle.get("counts")
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    rle_obj = {"counts": counts, "size": rle.get("size")}

    mask = maskUtils.decode(rle_obj)
    if mask.ndim == 3:
        mask = mask[..., 0]

    polygons = sv.mask_to_polygons(mask)
    return [p.flatten().tolist() for p in polygons]


def polygons_to_rle(polygons: list[list[float]], height: int, width: int) -> dict[str, Any]:
    """Convert polygon(s) to COCO RLE.

    Args:
        polygons: List of polygon coordinate lists in COCO format, where each polygon is a flat list of floats [x1,y1,x2,y2,...].
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        COCO RLE dict with 'size' ([height, width]) and 'counts' (str when returned).
        Counts are encoded as ASCII string for COCO compatibility.

    Note:
        Uses pycocotools to handle multiple contours and holes correctly. If multiple
        polygons are provided, they are merged into a single RLE.
    """
    # Use pycocotools to convert polygon(s) directly to RLE. This correctly handles
    # multiple contours and holes according to COCO conventions.
    # `frPyObjects` accepts a list of polygons (each a list of x,y coords) and
    # returns RLEs which we merge into a single RLE for the object.
    rles = maskUtils.frPyObjects(polygons, height, width)
    # frPyObjects may return a single RLE or a list; ensure we have a merged RLE
    if isinstance(rles, list):
        rle = maskUtils.merge(rles)
    else:
        rle = rles
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    rle["size"] = [height, width]
    return rle


def annotation_rle_to_polygon(annotation: dict[str, Any]) -> dict[str, Any]:
    """Mutate a COCO annotation dict to convert RLE segmentation to polygon segmentation.

    Args:
        annotation: COCO annotation dict with 'segmentation' field.

    Returns:
        The mutated annotation dict with 'segmentation' as a list of polygons.
        If already polygons, returns unchanged.

    Note:
        Modifies the annotation in-place. Sets 'iscrowd' to 0 for polygon format.
    """
    seg = annotation.get("segmentation")
    if isinstance(seg, dict):
        polygons = rle_to_polygons(seg)
        annotation["segmentation"] = polygons
        annotation["iscrowd"] = 0
    return annotation


def annotation_polygon_to_rle(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    """Mutate a COCO annotation dict to convert polygon segmentation to RLE.

    Args:
        annotation: COCO annotation dict with 'segmentation' field.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        The mutated annotation dict with 'segmentation' as an RLE dict.
        If already RLE, returns unchanged.

    Note:
        Modifies the annotation in-place. Sets 'iscrowd' to 1 for RLE format.
    """
    seg = annotation.get("segmentation")
    if isinstance(seg, list):
        rle = polygons_to_rle(seg, height, width)
        annotation["segmentation"] = rle
        annotation["iscrowd"] = 1
    return annotation


def convert_coco_mask_type(input_coco: str, output_json: str | None = None, to_format: str = "polygons") -> dict[str, Any]:
    """Convert segmentation encodings in a COCO annotation file and write a new JSON.

        Args:
            input_coco: Path to input COCO annotations JSON (or COCO object or dict).
            output_json: Path where converted JSON will be saved. If None, no file is written.
            to_format: Either 'polygons' or 'rle'.

    1    Returns:
            The converted COCO dataset dict.

        Example:
            >>> dataset = convert_coco_mask_type('annotations.json', 'converted.json', 'rle')
            # Converts all segmentations to RLE and saves to 'converted.json'
    """
    coco = load_coco(input_coco)
    dataset = coco.dataset

    images = dataset["images"]
    img_size_map = {img["id"]: (img.get("height"), img.get("width")) for img in images}

    annotations = dataset["annotations"]
    converted = []
    for ann in annotations:
        img_id = ann.get("image_id")
        h, w = img_size_map.get(img_id, (None, None))
        if to_format == "polygons":
            annotation_rle_to_polygon(ann)
        else:
            annotation_polygon_to_rle(ann, h, w)
        converted.append(ann)

    dataset["annotations"] = converted
    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(dataset, f)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert COCO segmentation encodings between RLE and polygon formats.")
    parser.add_argument("input_file", type=str, help="Path to input COCO annotations JSON")
    parser.add_argument("output_file", type=str, nargs="?", default=None, help="Path to write converted COCO annotations JSON (optional). If omitted no file will be written")
    parser.add_argument("--to", choices=["polygons", "rle"], default="polygons", help="Target segmentation format")
    args = parser.parse_args()

    convert_coco_mask_type(args.input_file, args.output_file, to_format=args.to)


if __name__ == "__main__":
    main()
