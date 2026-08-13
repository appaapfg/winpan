"""Utility to merge multiple COCO datasets into one.

This script provides a function to merge several COCO-formatted datasets (from files or dictionaries)
into a single dataset. It handles re-indexing of images, annotations, and licenses to ensure
uniqueness, while validating that categories and image file names do not conflict across datasets.

Key features:
- Merges 'images', 'annotations', and 'licenses' from multiple COCO sources.
- Re-indexes IDs for images and annotations to prevent collisions.
- Enforces category consistency: fails if a category ID or name is used inconsistently
  across datasets. Category IDs are NOT re-indexed.
- Enforces image uniqueness: fails if the same 'file_name' appears in multiple datasets.
- Can return the merged dataset as a Python dictionary and optionally write it to a JSON file.

Notes:
- Performance: Merging large datasets may consume significant memory due to loading all data into memory.
- Edge cases: Empty input lists raise ValueError; duplicate file_names or category conflicts halt the process.

Example CLI usage:
python -m panopticon.data_engine.merge_coco --inputs path/to/coco1.json path/to/coco2.json --output_file path/to/merged.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from tqdm import tqdm

from panopticon.data_engine.util import load_coco


def merge_coco_datasets(coco_inputs: list[str | dict], output_file: str | None = None) -> dict[str, Any]:
    """
    Merges multiple COCO datasets, re-indexing images and annotations.

    Args:
        coco_inputs: A list of paths (str) to COCO JSON files or pre-loaded COCO dictionaries.
        output_file: Optional path (str) to save the merged COCO JSON file.

    Returns:
        A dictionary representing the merged COCO dataset with keys: 'info', 'licenses', 'images', 'annotations', 'categories'.

    Raises:
        ValueError: If there are category conflicts (same ID with different names or vice-versa), image conflicts (duplicate file_names), or empty input list.

    Examples:
        >>> datasets = [{"images": [], "annotations": [], "categories": []}]
        >>> merged = merge_coco_datasets(datasets)
        >>> len(merged["images"])
        0
    """
    if not coco_inputs:
        raise ValueError("coco_inputs list cannot be empty.")

    datasets = [load_coco(inp).dataset for inp in coco_inputs]

    # 1. Merge categories and check for conflicts
    merged_categories = {}
    cat_name_to_id = {}
    for dataset in datasets:
        for category in dataset.get("categories", []):
            cat_id = category["id"]
            cat_name = category["name"]

            if cat_id in merged_categories and merged_categories[cat_id]["name"] != cat_name:
                raise ValueError(f"Category ID {cat_id} has conflicting names: " f"'{merged_categories[cat_id]['name']}' and '{cat_name}'.")
            if cat_name in cat_name_to_id and cat_name_to_id[cat_name] != cat_id:
                raise ValueError(f"Category name '{cat_name}' has conflicting IDs: " f"{cat_name_to_id[cat_name]} and {cat_id}.")

            if cat_id not in merged_categories:
                merged_categories[cat_id] = category
                cat_name_to_id[cat_name] = cat_id

    # 2. Initialize merged dataset structure
    merged_dataset = {
        "info": datasets[0].get("info", {}),
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": sorted(list(merged_categories.values()), key=lambda c: c["id"]),
    }

    # 3. Merge images, annotations, and licenses with re-indexing
    next_image_id = 1
    next_annotation_id = 1
    next_license_id = 1
    seen_file_names = set()

    for dataset in tqdm(datasets, desc="Merging datasets"):
        image_id_map = {}  # Old image ID -> new image ID for the current dataset
        license_id_map = {}  # Old license ID -> new license ID for the current dataset

        # Re-index licenses first to build the mapping
        for license_obj in dataset.get("licenses", []):
            old_license_id = license_obj["id"]
            new_license_id = next_license_id
            license_id_map[old_license_id] = new_license_id

            license_obj["id"] = new_license_id
            merged_dataset["licenses"].append(license_obj)
            next_license_id += 1

        # Re-index images
        # Also check for duplicate file_names to maintain uniqueness.
        for image in dataset.get("images", []):
            file_name = image["file_name"]
            if file_name in seen_file_names:
                raise ValueError(f"Duplicate image file_name found: {file_name}")
            seen_file_names.add(file_name)

            old_image_id = image["id"]
            new_image_id = next_image_id
            image_id_map[old_image_id] = new_image_id

            image["id"] = new_image_id
            # Update license_id if the image references one
            if "license_id" in image and image["license_id"] in license_id_map:
                image["license_id"] = license_id_map[image["license_id"]]
            merged_dataset["images"].append(image)
            next_image_id += 1

        # Re-index annotations
        for annotation in dataset.get("annotations", []):
            old_image_id = annotation["image_id"]
            if old_image_id in image_id_map:
                annotation["image_id"] = image_id_map[old_image_id]
                annotation["id"] = next_annotation_id
                merged_dataset["annotations"].append(annotation)
                next_annotation_id += 1

    if output_file:
        with open(output_file, "w") as f:
            json.dump(merged_dataset, f, indent=2)
        print(f"Merged dataset saved to {output_file}")

    return merged_dataset


def main():
    """CLI wrapper for merging COCO datasets."""
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO datasets into one.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to input COCO JSON files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the merged COCO JSON file. If not provided, prints summary.",
    )
    args = parser.parse_args()

    try:
        merged = merge_coco_datasets(args.inputs, args.output_file)
        print("\nMerge successful!")
        print(f"  - Total images: {len(merged['images'])}")
        print(f"  - Total annotations: {len(merged['annotations'])}")
        print(f"  - Total categories: {len(merged['categories'])}")
    except ValueError as e:
        print(f"\nError during merge: {e}")


if __name__ == "__main__":
    main()
