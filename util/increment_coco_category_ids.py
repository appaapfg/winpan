"""Utility to increment all category IDs by +1 in a COCO dataset and update annotations accordingly.

Description:
    This module provides functionality to update category IDs in a COCO-formatted dataset
    by incrementing each ID by +1, and correspondingly updating all annotation category_id
    references to maintain consistency.

Steps:
    1. Load a COCO annotation JSON.
    2. Increment each category's ID by +1.
    3. Update all annotations' category_id fields by +1.
    4. Save the updated dataset to a new file or overwrite the input.

Notes:
    - Assumes the input COCO JSON is valid.
    - Category IDs are incremented regardless of their original values.
    - Annotations with category_id not matching any category will still be incremented.

Example CLI usage:
    python util/increment_coco_category_ids.py --input_file path/to/annotations.coco.json --output_file path/to/updated.coco.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def increment_coco_category_ids(
    coco_input: str | dict,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Increment all category IDs by +1 in a COCO dataset and update annotations.

    Description:
        Loads a COCO dataset, increments each category ID by +1, and updates all
        annotation category_id fields accordingly to maintain referential integrity.

    Args:
        coco_input (str | dict): Path to COCO JSON file or loaded dataset dict.
        output_file (str | None, optional): Path to save updated dataset. If None, returns the dict.

    Returns:
        dict[str, Any]: Updated COCO dataset dict.
    """
    if isinstance(coco_input, str):
        with open(coco_input) as f:
            data = json.load(f)
    else:
        data = coco_input

    # Increment category IDs
    for cat in data.get("categories", []):
        cat["id"] += 1

    # Update annotation category_ids
    for ann in data.get("annotations", []):
        ann["category_id"] += 1

    if output_file:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    return data


def main():
    parser = argparse.ArgumentParser(description="Increment category IDs by +1 in a COCO dataset and update annotations.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input COCO annotation JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save updated COCO JSON")

    args = parser.parse_args()

    updated_data = increment_coco_category_ids(
        coco_input=args.input_file,
        output_file=args.output_file,
    )
    print(f"Updated {len(updated_data.get('categories', []))} categories and {len(updated_data.get('annotations', []))} annotations.")


if __name__ == "__main__":
    main()
