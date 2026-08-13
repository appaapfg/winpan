"""
Module for querying and pretty-printing COCO dataset image IDs containing a specific category.

This script loads a COCO annotation JSON file and finds all image IDs that contain
annotations for a given category name. It then prints the results in a formatted,
color-coded output for easy reading.

Example:
    python query_category_coco.py --coco_json annotations.json --category person

Args:
    --coco_json: Path to the COCO annotation JSON file.
    --category: Name of the category to search for (default: 'person').

Note:
    Requires the panopticon.data_engine.util.load_coco function to load the COCO data.
"""

import argparse

from panopticon.data_engine.util import load_coco

# ANSI colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"


def list_images_with_category(coco_json, category_name):
    """
    Finds all image IDs in a COCO dataset that contain a specific category.

    Args:
        coco_json (str): Path to the COCO annotation JSON file.
        category_name (str): Name of the category to search for.

    Returns:
        tuple: A tuple containing:
            - list: Sorted list of image IDs containing the category.
            - COCO: The loaded COCO object for further use.
    """
    coco = load_coco(coco_json)

    # Retrieve category IDs matching the provided name
    cat_ids = coco.getCatIds(catNms=[category_name])
    if not cat_ids:
        print(f"{C_RED}✘ Category '{category_name}' not found in dataset.{C_RESET}")
        return [], coco

    cat_id = cat_ids[0]

    # Find all annotation IDs associated with this category
    ann_ids = coco.getAnnIds(catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)

    # Extract unique image IDs from annotations and sort them
    img_ids = sorted({ann["image_id"] for ann in anns})

    return img_ids, coco


def pretty_print_results(category, img_ids, coco):
    """
    Pretty-prints the results of the category query in a formatted, color-coded output.

    Args:
        category (str): The category name that was searched for.
        img_ids (list): List of image IDs containing the category.
        coco (COCO): The loaded COCO object to retrieve image info.
    """
    print()
    # Print header with category name
    print(f"{C_BOLD}{C_BLUE}==============================")
    print(f" Images containing '{category}' ")
    print(f"=============================={C_RESET}\n")

    if not img_ids:
        print(f"{C_YELLOW}No images contain '{category}'.{C_RESET}")
        return

    print(f"{C_GREEN}✓ Found {len(img_ids)} images:{C_RESET}\n")

    # Print table header
    print(f"{C_BOLD}{'Image ID':<10}  File Name{C_RESET}")
    print(f"{'-' * 10}  {'-' * 40}")

    # Print each image ID and its file name
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        print(f"{img_id:<10}  {img_info.get('file_name', 'N/A')}")

    print()
    print(f"{C_BLUE}Done.{C_RESET}")


def main():
    parser = argparse.ArgumentParser(description="Pretty-print image IDs containing a given COCO category.")
    parser.add_argument("--coco_json", required=True, help="Path to COCO annotation JSON")
    parser.add_argument("--category", default="person", help="Category name (default: person)")
    args = parser.parse_args()

    img_ids, coco = list_images_with_category(args.coco_json, args.category)
    pretty_print_results(args.category, img_ids, coco)


if __name__ == "__main__":
    main()
