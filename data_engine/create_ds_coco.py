import argparse
import json
import os
from datetime import datetime

import datumaro as dm
from pycocotools.coco import COCO

# Constants for COCO metadata fields, updated dynamically
current_date = datetime.now().strftime("%Y/%m/%d")
current_year = datetime.now().year


def create_unlabeled_coco_dataset(image_directory: str, author_name: str, licence_name: str) -> tuple[COCO, str]:
    """Creates an unlabeled COCO dataset JSON sidecar for a directory of images.

    Args:
        image_directory (str): Path to the directory containing images.
        author_name (str): Name of the author/contributor for COCO metadata.
        licence_name (str): Name of the license for COCO metadata.

    Returns:
        tuple[COCO, str]: A tuple containing:
            - coco (COCO): The pycocotools COCO object loaded from the generated JSON file.
            - output_json_path (str): Absolute path to the generated COCO JSON file.

    Raises:
        FileNotFoundError: If the image directory does not exist.
        OSError: If file operations (e.g., renaming or writing JSON) fail.

    Note:
        This function uses Datumaro to import images and export to COCO format.
        It modifies the exported JSON to include custom metadata.
    """
    # Import images as a Datumaro dataset
    dataset = dm.Dataset.import_from(image_directory, format="image_dir")
    output_json_path = os.path.join(image_directory, "_annotations.coco.json")
    coco_info = {"contributor": author_name, "date_created": current_date, "version": "1", "year": current_year, "description": ""}

    coco_license = {"id": 0, "name": licence_name, "url": ""}  # No URL provided

    # Export the dataset to COCO instances format
    print(f"Exporting dataset to COCO format at: {output_json_path}")
    dataset.export(
        save_dir=f"{os.path.dirname(output_json_path)}",
        format="coco_instances",
    )

    # Rename the default export file and clean up temporary directories
    default_path = os.path.join(os.path.dirname(output_json_path), "annotations", "instances_default.json")
    os.rename(default_path, output_json_path)
    os.rmdir(os.path.join(os.path.dirname(output_json_path), "annotations"))

    print("Successfully created unlabeled COCO dataset.")

    print(f"Adding COCO metadata to: {output_json_path}")
    with open(output_json_path, "r+") as f:
        coco_data = json.load(f)

        # Add info
        coco_data["info"] = coco_info

        # Add licenses
        coco_data["licenses"] = [coco_license]

        # Assign license to all images
        for image in coco_data.get("images", []):
            image["license"] = coco_license["id"]

        # Overwrite file
        f.seek(0)
        json.dump(coco_data, f, indent=4)
        f.truncate()

    print("Successfully added COCO metadata.")
    coco = COCO(output_json_path)
    return coco, output_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an unlabeled COCO dataset JSON with adjustable author and licence metadata.")
    parser.add_argument("--image-dir", help="Path to directory containing images (default: %(default)s)")
    parser.add_argument("--author-name", default="Hellenic Army PvT (SC) Kallinteris Andreas", help="Author / contributor name to embed in COCO info")
    parser.add_argument("--license-name", default="Greek Army Classified", help="Licence name to embed in COCO metadata")
    args = parser.parse_args()

    dataset, result_path = create_unlabeled_coco_dataset(args.image_dir, args.author_name, args.license_name)
    print(f"COCO file written to: {result_path} (images: {len(dataset)})")
