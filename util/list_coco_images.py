"""A utility script to list all image filenames present in a COCO dataset JSON file.

Example usage:
    # List filenames from a COCO dataset
    python list_coco_images.py --input input.json
"""

import argparse

from pycocotools.coco import COCO

from panopticon.data_engine.util import load_coco


def list_image_filenames_from_coco(
    coco_input: str | COCO | dict,
) -> list[str]:
    """List all image filenames from a COCO dataset.

    Args:
        coco_input (str | COCO | dict): Path to COCO JSON file, loaded COCO object, or dataset dict.

    Returns:
        list[str]: A list of image filenames.
    """
    coco = load_coco(coco_input, silent=True)

    filenames: list[str] = []
    for img_meta in coco.dataset.get("images", []):
        if "file_name" in img_meta:
            filenames.append(img_meta["file_name"])
    return filenames


def main():
    parser = argparse.ArgumentParser(description="List image filenames from a COCO dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to COCO annotation JSON file.")

    args = parser.parse_args()

    filenames = list_image_filenames_from_coco(args.input)
    for filename in filenames:
        print(filename)
    # print(f"Total images found: {len(filenames)}")


if __name__ == "__main__":
    main()
