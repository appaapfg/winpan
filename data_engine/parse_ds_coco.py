"""Utilities for loading and processing COCO dataset images and annotations.

This module provides functions to load COCO-formatted datasets, filter annotations
by categories, and remap category IDs for consistent indexing starting from 0.
"""

import argparse
import json
import os
from typing import Any

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def load_coco_images_and_annotations(coco_or_path, category_filter: list[dict[str, Any]] | None = None, max_images: int | None = None) -> tuple[list[Image.Image], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Load images, per-image annotations, and used categories from a COCO dataset.

    Args:
        coco_or_path: Either a COCO object or a path to a .coco.json file.
        category_filter: Optional list[dict] of category dicts in COCO format (each with id, name, ...). If provided, only annotations whose category_id is in this set are kept.
        max_images: Optional[int]. If provided, the function will raise a ValueError if the number of images in the dataset exceeds this value.

    Behavior (per user spec):
     - Images are loaded relative to the JSON file directory (no override path).
     - Missing image files raise an error.
     - Images kept even if (after filtering) they have zero annotations.
     - Ordering of images is the original COCO image id order (coco.getImgIds()).
     - Categories returned are ONLY those referenced by (filtered) annotations.
     - Returned categories are re-indexed so that ids start from 0 (continuous); annotations' category_id fields are updated accordingly.
     - Raw annotations (all original fields) are preserved except for category_id remapping.
     - iscrowd annotations are NOT filtered out (kept as-is).

    Returns:
        List[PIL.Image.Image]: Loaded images in the order of coco.getImgIds().
        List[List[Dict[str, Any]]]: Annotations per image (after optional filtering) with remapped category_id.
        List[Dict[str, Any]]: Remapped category dicts actually referenced, ids starting at 0.
    """
    # Obtain COCO object
    if isinstance(coco_or_path, str):
        coco_path = coco_or_path
        coco = COCO(coco_path)
    else:
        coco = coco_or_path
        assert isinstance(coco, COCO), "Input must be a COCO object or path to a COCO json file"
        # Try to infer path for relative image loading (if available in coco.dataset)
        coco_path = getattr(coco, "annotation_file", None)

    if isinstance(coco_or_path, str):
        base_dir = os.path.dirname(os.path.abspath(coco_or_path))
    else:
        base_dir = os.path.dirname(os.path.abspath(coco_path)) if coco_path else os.getcwd()

    # Build allowed category id set (if category_filter provided)
    allowed_category_ids = None
    if category_filter is not None:
        allowed_category_ids = {c["id"] for c in category_filter if "id" in c}

    # Image IDs in original order
    img_ids = coco.getImgIds()

    if max_images is not None and len(img_ids) > max_images:
        raise ValueError(f"Number of images ({len(img_ids)}) exceeds max_images limit ({max_images}).")

    images_info = coco.loadImgs(img_ids)

    loaded_images: list[Image.Image] = []
    annotations_per_image: list[list[dict[str, Any]]] = []

    iterator = tqdm(enumerate(images_info), total=len(images_info), desc="Loading COCO images & annotations")

    for idx, img_info in iterator:
        file_name = img_info.get("file_name")
        img_path = os.path.join(base_dir, file_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        loaded_images.append(image)
        ann_ids = coco.getAnnIds(imgIds=[img_info["id"]])
        anns = coco.loadAnns(ann_ids)
        if allowed_category_ids is not None:
            anns = [a for a in anns if a.get("category_id") in allowed_category_ids]
        annotations_per_image.append(anns)

    # Collect category ids actually used after filtering
    used_category_ids = set()
    for ann_list in annotations_per_image:
        for ann in ann_list:
            cid = ann.get("category_id")
            if cid is not None:
                used_category_ids.add(cid)

    # Original categories dict by id
    original_categories_by_id = {}
    for cat in coco.dataset.get("categories", []):
        if "id" in cat:
            original_categories_by_id[cat["id"]] = cat

    # Build kept categories preserving original order appearance in coco.dataset
    kept_categories_original_order = [c for c in coco.dataset.get("categories", []) if c.get("id") in used_category_ids]

    # Remap category ids starting from 0 for continuous indexing
    old_to_new = {}
    new_categories = []
    for new_id, cat in enumerate(kept_categories_original_order):
        old_id = cat["id"]
        old_to_new[old_id] = new_id
        # Shallow copy to avoid mutating original
        cat_new = dict(cat)
        cat_new["id"] = new_id
        new_categories.append(cat_new)

    # Update annotations' category_id with new ids
    for ann_list in annotations_per_image:
        for ann in ann_list:
            old_id = ann.get("category_id")
            if old_id in old_to_new:
                ann["category_id"] = old_to_new[old_id]

    return loaded_images, annotations_per_image, new_categories


def _cli():
    parser = argparse.ArgumentParser(description="Load COCO images & annotations and print summary.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input COCO .coco.json file")
    parser.add_argument("--categories_json", type=str, default=None, help="Optional path to a JSON file containing a list[dict] of category objects to filter by.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images allowed in the dataset. If exceeded, the function will raise an error.")
    # tqdm always enabled (argument removed per update)
    args = parser.parse_args()

    category_filter = None
    if args.categories_json:
        with open(args.categories_json) as f:
            category_filter = json.load(f)
            if not isinstance(category_filter, list):
                raise ValueError("categories_json must contain a JSON list of category dicts")

    images, anns_per_img, cats = load_coco_images_and_annotations(args.input_file, category_filter=category_filter, max_images=args.max_images)

    # Print all returned values
    print("\n--- Images (filenames) ---")
    for i, img in enumerate(images):
        fname = getattr(img, "filename", None)
        if fname:
            print(f"[{i}] {fname}")
        else:
            print(f"[{i}] <PIL.Image.Image object>")

    print("\n--- Annotations per image ---")
    for i, ann_list in enumerate(anns_per_img):
        print(f"Image {i} annotations:")
        for ann in ann_list:
            print(f"  {ann}")

    print("\n--- Categories (remapped) ---")
    for cat in cats:
        print(cat)


if __name__ == "__main__":
    _cli()
