"""A utility script to remove specified categories and their associated annotations from a COCO dataset JSON file.

This script provides functionality to filter COCO datasets by category, either by name or ID,
and optionally remove images that no longer have any annotations after the filtering process.

Example usage:
    # Remove categories 'person' and 'car' from a dataset
    python remove_category.py --input input.json --categories person,car --output output.json

    # Remove category with ID 17 and drop images without annotations
    python remove_category.py --input input.json -c 17 --drop-empty-images -o output.json
"""

import argparse
import json
import os
from collections.abc import Iterable, Sequence

from pycocotools.coco import COCO


def _to_category_id_set(dataset: dict, categories: Sequence[str | int]) -> set[int]:
    """Map input category identifiers (names or ids) to a set of category ids.

    If an element is an int, it's treated as a category id.
    If a string, it's treated as a category name and looked up in dataset["categories"].
    Unknown names are ignored with a warning.

    Args:
        dataset (dict): The COCO dataset dictionary.
        categories (Sequence[str | int]): An iterable of category names (str) or IDs (int).

    Returns:
        set[int]: A set of unique integer category IDs.
    """
    name_to_id = {c.get("name"): c.get("id") for c in dataset.get("categories", [])}
    ids: set[int] = set()
    for c in categories:
        if isinstance(c, int):
            ids.add(c)
        elif isinstance(c, str):
            if c in name_to_id and name_to_id[c] is not None:
                ids.add(int(name_to_id[c]))
            else:
                print(f"[remove-category] Warning: category name '{c}' not found. Skipping.")
        else:
            print(f"[remove-category] Warning: unsupported category identifier type: {type(c)}. Skipping.")
    return ids


def remove_categories_from_coco(
    coco_or_path: COCO | dict | str,
    categories_to_remove: Iterable[str | int],
    output_file: str | None = None,
    remove_images_without_annotations: bool = False,
) -> COCO | dict:
    """Remove categories and all associated annotations from a COCO dataset.

    Args:
        coco_or_path: A COCO object, a loaded COCO dict, or a path to a COCO JSON file.
        categories_to_remove: Iterable of category names or ids to remove.
        output_file: Optional path to write the updated dataset JSON.
        remove_images_without_annotations: If True, drop images that end up with zero annotations.

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
    dataset.setdefault("images", [])
    dataset.setdefault("annotations", [])
    dataset.setdefault("categories", [])

    # Determine ids to remove
    ids_to_remove = _to_category_id_set(dataset, list(categories_to_remove))
    if not ids_to_remove:
        print("[remove-category] Nothing to remove (no matching category ids). Returning unchanged dataset.")
        return coco_obj or dataset

    # Filter categories
    old_cat_count = len(dataset["categories"])
    dataset["categories"] = [c for c in dataset["categories"] if int(c.get("id", -1)) not in ids_to_remove]
    new_cat_count = len(dataset["categories"])

    # Filter annotations that reference removed categories
    old_ann_count = len(dataset["annotations"])
    dataset["annotations"] = [a for a in dataset["annotations"] if int(a.get("category_id", -1)) not in ids_to_remove]
    new_ann_count = len(dataset["annotations"])

    # Optionally filter images that now have zero annotations
    removed_images = 0
    if remove_images_without_annotations:
        ann_img_ids = {a.get("image_id") for a in dataset["annotations"]}
        old_img_count = len(dataset["images"])
        dataset["images"] = [img for img in dataset["images"] if img.get("id") in ann_img_ids]
        removed_images = old_img_count - len(dataset["images"])

    # Persist if requested
    if output_file and output_file.lower() != "none":
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(dataset, f)

    removed_cats = old_cat_count - new_cat_count
    removed_anns = old_ann_count - new_ann_count
    print(f"[remove-category] Done. Removed categories: {removed_cats}, removed annotations: {removed_anns}" + (f", removed images: {removed_images}" if remove_images_without_annotations else ""))

    # Reindex if we have a COCO object
    if coco_obj is not None:
        coco_obj.dataset = dataset
        coco_obj.createIndex()
        return coco_obj
    return dataset


def _parse_categories_arg(values: list[str]) -> list[str | int]:
    """Parse --categories input supporting comma-separated lists and repeated flags.

    Args:
        values (list[str]): A list of strings, where each string can be a comma-separated
            list of category names or IDs, or a single category name/ID.

    Returns:
        list[str | int]: A list of parsed category identifiers, which can be either
            strings (names) or integers (IDs).
    """
    result: list[str | int] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(description="Remove categories and their annotations from a COCO dataset.")
    parser.add_argument("--input", required=True, help="Path to input COCO JSON file.")
    parser.add_argument(
        "--categories",
        "-c",
        action="append",
        required=True,
        help="Category names or ids to remove. Can be repeated or comma-separated. Example: -c person,car -c 17",
    )
    parser.add_argument("--output", "-o", default=None, help="Optional path to write the updated COCO JSON.")
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="If set, remove images that end up with zero annotations after filtering.",
    )

    args = parser.parse_args()
    cats = _parse_categories_arg(args.categories)

    # Load as dict to avoid requiring pycocotools indexing for simple remove
    with open(args.input) as f:
        dataset = json.load(f)

    remove_categories_from_coco(
        dataset,
        cats,
        output_file=args.output,
        remove_images_without_annotations=args.drop_empty_images,
    )


if __name__ == "__main__":
    main()
