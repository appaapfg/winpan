#!/usr/bin/env python3
"""
Pretty-print all COCO image IDs (and their JSON index positions)
that contain a specific category (e.g., 'bullozer').
"""

import argparse
from panopticon.data_engine.util import load_coco
from pycocotools.coco import COCO

# ANSI colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"


def list_images_with_category(coco_json, category_name):
    coco = load_coco(coco_json)

    # Build mapping: image_id → index in coco.dataset["images"]
    image_index_map = {
        img["id"]: idx for idx, img in enumerate(coco.dataset.get("images", []))
    }

    # Find category
    cat_ids = coco.getCatIds(catNms=[category_name])
    if not cat_ids:
        print(f"{C_RED}✘ Category '{category_name}' not found in dataset.{C_RESET}")
        return [], coco, image_index_map

    cat_id = cat_ids[0]

    # Find annotations with this category
    ann_ids = coco.getAnnIds(catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)

    img_ids = sorted({ann["image_id"] for ann in anns})

    return img_ids, coco, image_index_map


def pretty_print_results(category, img_ids, coco, image_index_map):
    print()
    print(f"{C_BOLD}{C_BLUE}==============================")
    print(f" Images containing '{category}' ")
    print(f"=============================={C_RESET}\n")

    if not img_ids:
        print(f"{C_YELLOW}No images contain '{category}'.{C_RESET}")
        return

    print(f"{C_GREEN}✓ Found {len(img_ids)} images:{C_RESET}\n")

    print(f"{C_BOLD}{'JSON Idx':<8} {'Image ID':<10}  File Name{C_RESET}")
    print(f"{'-'*8} {'-'*10}  {'-'*40}")

    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        json_index = image_index_map.get(img_id, -1)

        print(f"{json_index:<8} {img_id:<10}  {img_info.get('file_name', 'N/A')}")

    print()
    print(f"{C_BLUE}Done.{C_RESET}")


def main():
    parser = argparse.ArgumentParser(description="Pretty-print image IDs containing a given COCO category.")
    parser.add_argument("--coco_json", required=True, help="Path to COCO annotation JSON")
    parser.add_argument("--category", default="person", help="Category name (default: person)")
    args = parser.parse_args()

    img_ids, coco, image_index_map = list_images_with_category(args.coco_json, args.category)
    pretty_print_results(args.category, img_ids, coco, image_index_map)


if __name__ == "__main__":
    main()
