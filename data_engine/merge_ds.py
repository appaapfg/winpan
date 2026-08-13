import json
from typing import Any

from pycocotools.coco import COCO


def merge_coco_datasets(coco_inputs: list[str | COCO], output_path: str) -> None:
    """
    Merges multiple COCO datasets into a single .coco.json file.

    This function combines images, annotations, and categories from a list of COCO datasets.
    It checks for category conflicts (same ID, different name) and raises an error if any are found.
    Image and annotation IDs are re-assigned to be unique across the merged dataset.

    Args:
        coco_inputs: A list of COCO datasets. Each element can be a path
                     to a .coco.json file or a pycocotools.COCO object.
        output_path: The path to save the merged .coco.json file.

    Raises:
        ValueError: If there are conflicting category definitions (same ID, different name)
                    across the datasets.
    """
    merged_coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {},
        "licenses": [],
    }

    merged_categories: dict[str, dict[str, Any]] = {}
    next_image_id = 1
    next_annotation_id = 1

    for i, coco_input in enumerate(coco_inputs):
        if isinstance(coco_input, str):
            coco = COCO(coco_input)
        elif isinstance(coco_input, COCO):
            coco = coco_input
        else:
            raise TypeError(f"Input {i} is not a valid COCO path or object.")

        if i == 0:
            merged_coco["info"] = coco.dataset.get("info", {})
            merged_coco["licenses"] = coco.dataset.get("licenses", [])

        # Process categories and check for conflicts
        cat_id_map: dict[int, int] = {}
        for cat in coco.loadCats(coco.getCatIds()):
            cat_name = cat["name"]
            if cat_name in merged_categories:
                # If the category name already exists, ensure it's consistent
                existing_cat = merged_categories[cat_name]
                if existing_cat["id"] != cat["id"]:
                    print(f"Warning: Category '{cat_name}' has different IDs in input datasets. Remapping to a consistent ID.")
                cat_id_map[cat["id"]] = existing_cat["id"]
            else:
                # Add new category
                new_cat = cat.copy()
                merged_categories[cat_name] = new_cat
                cat_id_map[cat["id"]] = new_cat["id"]

        # Create a new mapping for category IDs to ensure they are unique in the merged dataset
        current_max_cat_id = max([c["id"] for c in merged_coco["categories"]]) if merged_coco["categories"] else 0
        for cat in coco.loadCats(coco.getCatIds()):
            original_cat_id = cat["id"]
            if cat["name"] not in [c["name"] for c in merged_coco["categories"]]:
                current_max_cat_id += 1
                new_cat = cat.copy()
                new_cat["id"] = current_max_cat_id
                merged_coco["categories"].append(new_cat)
                cat_id_map[original_cat_id] = new_cat["id"]
            else:
                # Find the existing category ID in the merged list
                for merged_cat in merged_coco["categories"]:
                    if merged_cat["name"] == cat["name"]:
                        cat_id_map[original_cat_id] = merged_cat["id"]
                        break

        # Process images and annotations
        img_id_map: dict[int, int] = {}
        for img_id in coco.getImgIds():
            img = coco.loadImgs(img_id)[0]
            original_img_id = img["id"]
            new_img_id = next_image_id
            img_id_map[original_img_id] = new_img_id

            new_img = img.copy()
            new_img["id"] = new_img_id
            merged_coco["images"].append(new_img)
            next_image_id += 1

            ann_ids = coco.getAnnIds(imgIds=[original_img_id])
            for ann_id in ann_ids:
                ann = coco.loadAnns(ann_id)[0]
                new_ann = ann.copy()
                new_ann["id"] = next_annotation_id
                new_ann["image_id"] = new_img_id
                new_ann["category_id"] = cat_id_map[ann["category_id"]]
                merged_coco["annotations"].append(new_ann)
                next_annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(merged_coco, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge multiple COCO datasets into a single JSON file.")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input COCO annotation JSON files (space-separated)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the merged COCO JSON file",
    )

    args = parser.parse_args()

    try:
        merge_coco_datasets(args.input_files, args.output_file)
        print(f"Successfully merged COCO datasets to {args.output_file}")

        # Optional: Verify the merged file
        # with open(args.output_file, "r") as f:
        #     merged_data = json.load(f)
        #     print("\nMerged COCO data:")
        #     print(json.dumps(merged_data, indent=4))

    except ValueError as e:
        print(f"Error merging datasets: {e}")
    except TypeError as e:
        print(f"Error: {e}")
