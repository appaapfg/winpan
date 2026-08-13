"""Utility to remove all segmentation masks from a COCO dataset.

This function removes the 'segmentation' field from all annotations in a COCO dataset,
effectively stripping out all segmentation masks while preserving bounding boxes and other metadata.
"""

import argparse
import json
import os

from pycocotools.coco import COCO


def remove_segmentation_masks_from_coco(
    coco_or_path: COCO | dict | str,
    output_file: str | None = None,
) -> COCO | dict:
    """Remove all segmentation masks from annotations in a COCO dataset.

    This function iterates through all annotations in the dataset and removes the 'segmentation'
    field if present, effectively stripping out segmentation masks while preserving bounding
    boxes and other annotation metadata.

    Args:
        coco_or_path: A COCO object, a loaded COCO dict, or a path to a COCO JSON file.
        output_file: Optional path to write the updated dataset JSON.

    Returns:
        If input is a COCO object, returns the same COCO object updated in-place (and re-indexed).
        Otherwise, returns the updated dataset dict.
    """
    # Load dataset from source
    coco_obj: COCO | None = None
    if isinstance(coco_or_path, str):
        with open(coco_or_path) as f:
            dataset = json.load(f)
    elif isinstance(coco_or_path, COCO):
        coco_obj = coco_or_path
        dataset = coco_obj.dataset
    elif isinstance(coco_or_path, dict):
        dataset = coco_or_path
    else:
        raise TypeError("coco_or_path must be a str path, COCO instance, or dataset dict")

    # Basic schema presence
    dataset.setdefault("annotations", [])

    # Count masks before removal
    masks_removed = 0
    for ann in dataset["annotations"]:
        if "segmentation" in ann:
            del ann["segmentation"]
            masks_removed += 1

    # Persist if requested
    if output_file and output_file.lower() != "none":
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(dataset, f)

    print(f"[remove-segmentation-masks] Done. Removed segmentation masks from {masks_removed} annotations.")

    # Reindex if we have a COCO object
    if coco_obj is not None:
        coco_obj.dataset = dataset
        coco_obj.createIndex()
        return coco_obj
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Remove all segmentation masks from a COCO dataset.")
    parser.add_argument("--input", required=True, help="Path to input COCO JSON file.")
    parser.add_argument("--output", "-o", default=None, help="Optional path to write the updated COCO JSON.")

    args = parser.parse_args()

    # Load as dict
    with open(args.input) as f:
        dataset = json.load(f)

    remove_segmentation_masks_from_coco(
        dataset,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
