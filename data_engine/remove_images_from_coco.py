"""Utility to remove specified images (by image_id) from a COCO dataset.

Description:
    This module provides functionality to filter out specific images and their associated
    annotations from a COCO dataset. The process involves loading the dataset, removing
    the specified images, and cleaning up related annotations while preserving the overall
    COCO structure.

    Steps:
    1. Load a COCO annotation JSON (images + existing annotations).
    2. Filter out images with specified image_ids.
    3. Filter out annotations associated with those image_ids.
    4. Return the modified COCO dataset as a dict.

Notes:
    - The function preserves the original COCO structure but with specified images and their annotations removed.
    - Categories are not cleaned up (unused categories remain).
    - The returned dict can be saved to a new JSON file or used directly.

Examples:
    python data_engine/remove_images_from_coco.py --coco_json path/to/annotations.coco.json --image_ids 1 2 3 --output path/to/modified.coco.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

from panopticon.data_engine.util import load_coco


def remove_images_from_coco(
    coco_input: str | dict,
    image_ids_to_remove: list[int],
) -> dict[str, Any]:
    """Remove specified images (by image_id) from a COCO dataset.

    This function loads a COCO dataset, filters out images with the specified image_ids,
    and removes all annotations associated with those images. The modified dataset is
    returned as a dictionary in COCO format.

    Args:
        coco_input: COCO file path, or dict containing COCO data.
        image_ids_to_remove: List of image_ids to remove from the dataset.

    Returns:
        Modified COCO dataset as a dict with specified images and their annotations removed.

    Raises:
        ValueError: If any image_id in image_ids_to_remove is not found in the dataset.
    """
    coco = load_coco(coco_input)

    # Get the original dataset dict
    if isinstance(coco_input, dict):
        coco_dict = coco_input.copy()
    else:
        coco_dict = cast(dict[str, Any], coco.dataset)

    # Validate that all image_ids exist
    existing_image_ids = {img["id"] for img in coco_dict.get("images", [])}
    missing_ids = set(image_ids_to_remove) - existing_image_ids
    if missing_ids:
        raise ValueError(f"Image IDs not found in dataset: {list(missing_ids)}")

    # Filter out images
    filtered_images = [img for img in coco_dict.get("images", []) if img["id"] not in image_ids_to_remove]

    # Filter out annotations associated with removed images
    filtered_annotations = [ann for ann in coco_dict.get("annotations", []) if ann["image_id"] not in image_ids_to_remove]

    # Create modified dataset
    modified_coco_dict = coco_dict.copy()
    modified_coco_dict["images"] = filtered_images
    modified_coco_dict["annotations"] = filtered_annotations

    return modified_coco_dict


def main():  # CLI wrapper
    parser = argparse.ArgumentParser(description="Remove specified images from a COCO dataset.")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO annotation JSON")
    parser.add_argument("--image_ids", type=int, nargs="+", required=True, help="List of image_ids to remove")
    parser.add_argument("--output", type=str, help="Path to save modified COCO JSON (optional)")

    args = parser.parse_args()

    try:
        modified_coco = remove_images_from_coco(
            coco_input=args.coco_json,
            image_ids_to_remove=args.image_ids,
        )

        print(f"Removed {len(args.image_ids)} images from dataset.")
        print(f"Remaining images: {len(modified_coco.get('images', []))}")
        print(f"Remaining annotations: {len(modified_coco.get('annotations', []))}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(modified_coco, f, indent=2)
            print(f"Modified dataset saved to: {args.output}")
        else:
            print("Modified dataset (first 3 images shown):")
            print(json.dumps(modified_coco.get("images", [])[:3], indent=2))

    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
