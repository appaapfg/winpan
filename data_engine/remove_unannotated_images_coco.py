"""Utility to remove images from a COCO dataset that have no annotations.

This script processes a COCO annotation JSON file and creates a new version
containing only images that have at least one annotation. The original images
are not deleted; only their references are removed from the dataset.

Steps:
1. Load the COCO annotation JSON.
2. Identify images that have annotations.
3. Create a new COCO dataset with only those images and their annotations.
4. Save the filtered dataset to a new JSON file.


Examples:
    Basic usage via CLI:

    .. code-block:: bash

        python data_engine/remove_unannotated_images_coco.py \
            --input_file path/to/annotations.coco.json \
            --output_file path/to/filtered_annotations.coco.json

    Programmatic usage:

    .. code-block:: python

        from panopticon.data_engine.remove_unannotated_images_coco import remove_unannotated_images_coco
        filtered = remove_unannotated_images_coco('path/to/annotations.coco.json')

Raises:
    FileNotFoundError: If the input file path does not exist.
    json.JSONDecodeError: If the input JSON is malformed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from pycocotools.coco import COCO

from panopticon.data_engine.util import load_coco


def remove_unannotated_images_coco(coco_input: str | COCO | dict) -> dict[str, Any]:
    """Remove images from a COCO dataset that have no annotations.

    Args:
        coco_input (str | COCO | dict): Path to COCO JSON file (as string), COCO object, or pre-loaded dict containing COCO annotations.

    Returns:
        dict[str, Any]: A new COCO dataset dict containing only images with annotations, mirroring the structure of the input COCO format.

    Raises:
        FileNotFoundError: If the input file path does not exist.
        json.JSONDecodeError: If the input JSON is malformed.
        ValueError: If the input is not a valid COCO format or type.
    """
    coco = load_coco(coco_input)

    # Collect all image_ids that have annotations
    annotated_image_ids = set()
    for ann in coco.dataset.get("annotations", []):
        annotated_image_ids.add(ann["image_id"])

    # Filter images to only those with annotations
    filtered_images = [img for img in coco.dataset["images"] if img["id"] in annotated_image_ids]

    # Filter annotations to only those belonging to the filtered images
    filtered_annotations = [ann for ann in coco.dataset.get("annotations", []) if ann["image_id"] in annotated_image_ids]

    # Create new dataset dict
    new_dataset = coco.dataset.copy()
    new_dataset["images"] = filtered_images
    new_dataset["annotations"] = filtered_annotations

    return new_dataset


def main():
    parser = argparse.ArgumentParser(description="Remove images from a COCO dataset that have no annotations.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input COCO annotation JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output filtered COCO annotation JSON file")

    args = parser.parse_args()

    # Load and process the dataset
    filtered_dataset = remove_unannotated_images_coco(args.input_file)

    # Save the filtered dataset
    with open(args.output_file, "w") as f:
        json.dump(filtered_dataset, f, indent=2)

    print(f"Filtered dataset saved to {args.output_file}")
    print(f"Original images: {len(load_coco(args.input_file).dataset['images'])}")
    print(f"Filtered images: {len(filtered_dataset['images'])}")
    print(f"Annotations: {len(filtered_dataset['annotations'])}")


if __name__ == "__main__":
    main()
