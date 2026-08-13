"""Utility to copy images from a COCO dataset to a new directory.

Description:
    This script reads a COCO-formatted JSON file to get a list of image
    file names. It then copies these images from a specified source
    directory to a destination directory.

Features:
    - Parses COCO JSON to identify all referenced images.
    - Copies files from a source to a destination directory.
    - Overwrites existing files in the destination directory.
    - Displays a progress bar using tqdm.
    - Requires the destination directory to exist.

Example CLI usage:
    python panopticon/util/copy_coco_images.py \\
        --coco_json /path/to/your/annotations.coco.json \\
        --source_dir /path/to/original/images \\
        --destination_dir /path/to/copied/images
"""

import argparse
import os
import shutil

from pycocotools.coco import COCO
from tqdm import tqdm


def copy_coco_images(coco_json_path: str, source_dir: str, destination_dir: str):
    """
    Copies images listed in a COCO JSON file from a source to a destination directory.

    Args:
        coco_json_path (str): Path to the COCO annotation JSON file.
        source_dir (str): The directory where the original images are stored.
        destination_dir (str): The directory where images will be copied.

    Raises:
        FileNotFoundError: If the coco_json_path, source_dir, or destination_dir do not exist.
        NotADirectoryError: If source_dir or destination_dir are not directories.
    """
    if not os.path.exists(coco_json_path):
        raise FileNotFoundError(f"COCO JSON file not found at: {coco_json_path}")
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Source directory not found or is not a directory: {source_dir}")
    if not os.path.isdir(destination_dir):
        raise NotADirectoryError(f"Destination directory not found or is not a directory: {destination_dir}")

    # Load the COCO dataset
    coco = COCO(coco_json_path)

    # Get all image IDs and load their metadata
    image_ids = coco.getImgIds()
    images_meta = coco.loadImgs(image_ids)

    if not images_meta:
        print("No images found in the COCO file.")
        return

    print(f"Found {len(images_meta)} images to copy from '{source_dir}' to '{destination_dir}'.")

    # Loop through all images and copy them
    copied_count = 0
    skipped_count = 0
    for image_info in tqdm(images_meta, desc="Copying images"):
        file_name = image_info["file_name"]
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)

        if os.path.exists(source_path):
            # shutil.copy will overwrite the destination file if it exists
            shutil.copy(source_path, destination_path)
            copied_count += 1
        else:
            print(f"Warning: Source file not found, skipping: {source_path}")
            skipped_count += 1

    print(f"\\nDone. Copied {copied_count} files. Skipped {skipped_count} files.")


def main():
    """CLI wrapper for the script."""
    parser = argparse.ArgumentParser(description="Copy images from a COCO dataset to a new directory.")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to the COCO annotation JSON file.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the original source images.")
    parser.add_argument("--destination_dir", type=str, required=True, help="Directory where the images will be copied. Must exist.")

    args = parser.parse_args()

    try:
        copy_coco_images(args.coco_json, args.source_dir, args.destination_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
