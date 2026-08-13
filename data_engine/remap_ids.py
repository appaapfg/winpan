"""Module for remapping category IDs in COCO datasets.

This module provides utilities to transform category IDs, names, and supercategories
in COCO-formatted annotation files, while updating associated annotations.

Overview:
    The primary function, remap_category_ids, allows remapping of category IDs
    based on a provided mapping dictionary. It handles validation, collision
    detection, and annotation updates.

Usage:
    Use the command-line interface via main() or import functions directly.

Dependencies:
    - pycocotools for COCO object handling.
    - panopticon.data_engine.util for loading COCO data.
"""

import argparse
import copy
import json
import os
from typing import Any

from panopticon.data_engine.util import load_coco


def _load_category_mapping(mapping_arg: str) -> dict[int, dict[str, Any]]:
    """Load category mapping from a JSON string or file path.

    Args:
        mapping_arg (str): Path to a JSON file or a JSON string containing the mapping.

    Returns:
        dict[int, dict[str, Any]]: Mapping from old category IDs to new category dicts,
            where each dict contains 'id' (int), 'name' (str), and 'supercategory' (str).

    Raises:
        ValueError: If keys cannot be converted to int, values are not dicts,
            or required keys are missing/invalid.

    Note:
        Expected structure:
        {
          "1": {"id": 10, "name": "new_name", "supercategory": "new_super"},
          "5": {"id": 20, "name": "other", "supercategory": "sup"}
        }
        Keys must be strings coercible to int. Values must only contain 'id', 'name',
        'supercategory' keys.
    """
    if os.path.exists(mapping_arg):
        with open(mapping_arg) as f:
            data = json.load(f)
    else:
        data = json.loads(mapping_arg)

    mapping: dict[int, dict[str, Any]] = {}
    for k, v in data.items():
        try:
            old_id = int(k)
        except ValueError:
            raise ValueError(f"Mapping key '{k}' is not an int")
        if not isinstance(v, dict):
            raise ValueError(f"Mapping value for {k} must be a dict")
        req_keys = {"id", "name", "supercategory"}
        present = set(v.keys())
        if present != req_keys:
            raise ValueError(f"Mapping for {k} must contain only keys {req_keys}, got {present}")
        if not isinstance(v["id"], int):
            raise ValueError(f"New category id for {k} must be int")
        if not isinstance(v["name"], str) or not isinstance(v["supercategory"], str):
            raise ValueError(f"New category name/supercategory for {k} must be strings")
        mapping[old_id] = {"id": v["id"], "name": v["name"], "supercategory": v["supercategory"]}
    return mapping


def remap_category_ids(input_dataset, category_mapping: dict[int, dict[str, Any]], allow_collisions: bool = False, output_file: str | None = None):
    """Remap category IDs (and their name + supercategory) in a COCO dataset.

    Parameters
    ----------
    input_dataset : str | COCO
        Path to COCO annotation JSON OR an existing COCO object.
    category_mapping : dict[int, dict]
        Mapping old_id -> new_category_dict with ONLY keys {id, name, supercategory}.
    allow_collisions : bool
        If False, raise ValueError when multiple old IDs map to the same new id.
        If True, allow multiple new category entries with identical id (may yield duplicate IDs).
    output_file : str | None
        If provided and not 'None', write the updated dataset JSON to this path.

    Behavior
    --------
    - Each old category id in mapping is REMOVED and REPLACED by its mapped new category dict.
    - Annotations referencing the old id are updated to the new id.
    - If an annotation has a 'category_str' field it is updated to the new name.
    - Categories not mentioned in mapping are left unchanged.
    - If any new id collides with an existing (unmapped) category id -> always error (regardless of allow_collisions).
    - If duplicates among new ids occur: allowed only when allow_collisions=True, otherwise error.
    - Note: When allow_collisions=True and duplicate ids are produced, the returned COCO object's internal index (cats) will only retain the last category per duplicated id due to pycocotools limitations, even though the raw dataset["categories"] preserves duplicates.

    Returns
    -------
    COCO : a new COCO object with remapped categories (original input COCO object is not mutated).
    """
    coco = load_coco(input_dataset)
    # Deep copy to ensure original COCO object remains unchanged
    new_coco = copy.deepcopy(coco)
    dataset = new_coco.dataset  # shallow reference inside deep copy

    if "categories" not in dataset:
        raise ValueError("Dataset missing 'categories'")
    if "annotations" not in dataset:
        raise ValueError("Dataset missing 'annotations'")

    # Extract old IDs from mapping for validation
    old_ids = set(category_mapping.keys())
    all_existing_ids = {c["id"] for c in dataset["categories"]}
    missing = old_ids - all_existing_ids
    if missing:
        raise ValueError(f"Old category ids not found in dataset: {sorted(missing)}")

    # Validate collisions with existing unmapped categories
    new_ids = [v["id"] for v in category_mapping.values()]
    unmapped_existing_ids = all_existing_ids - old_ids
    conflicts_with_unmapped = set(new_ids) & unmapped_existing_ids
    if conflicts_with_unmapped:
        raise ValueError(f"New ids {sorted(conflicts_with_unmapped)} already exist among untouched categories (disallowed)")

    # Validate duplicates among new IDs if collisions not allowed
    if not allow_collisions:
        dups = {nid for nid in new_ids if new_ids.count(nid) > 1}
        if dups:
            raise ValueError(f"Duplicate target ids {sorted(dups)} detected while allow_collisions is False")

    # Build lookup for mapped categories
    # Remove old categories
    remaining_categories = [c for c in dataset["categories"] if c["id"] not in old_ids]
    # Append new categories (order not important per user)
    for old_id, new_cat in category_mapping.items():
        # We only keep id, name, supercategory exactly
        remaining_categories.append({"id": new_cat["id"], "name": new_cat["name"], "supercategory": new_cat["supercategory"]})
    # Assign back
    dataset["categories"] = remaining_categories

    # Update annotations
    old_to_new = {old_id: new_cat["id"] for old_id, new_cat in category_mapping.items()}
    old_to_new_name = {old_id: new_cat["name"] for old_id, new_cat in category_mapping.items()}
    for ann in dataset["annotations"]:
        cid = ann.get("category_id")
        if cid in old_to_new:
            ann["category_id"] = old_to_new[cid]
            if "category_str" in ann:
                ann["category_str"] = old_to_new_name[cid]

    # Rebuild COCO indices after category changes
    new_coco.createIndex()

    # Optional save
    if output_file is not None and str(output_file).lower() != "none":
        with open(output_file, "w") as f:
            json.dump(dataset, f)

    return new_coco


def main():
    parser = argparse.ArgumentParser(description="Remap category IDs (and names/supercategories) in a COCO dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input COCO annotation JSON file.")
    parser.add_argument("--category_mapping", type=str, required=True, help="Path to JSON file OR JSON string specifying mapping old_id -> {id,name,supercategory}.")
    parser.add_argument("--output_file", type=str, default="None", help="Where to write updated dataset (optional).")
    parser.add_argument("--allow_collisions", action="store_true", help="Allow multiple old categories to map to same new id (duplicate ids).")

    args = parser.parse_args()
    category_mapping = _load_category_mapping(args.category_mapping)
    remap_category_ids(args.input_file, category_mapping, allow_collisions=args.allow_collisions, output_file=args.output_file)


if __name__ == "__main__":
    main()
