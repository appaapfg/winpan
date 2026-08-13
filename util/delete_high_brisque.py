import argparse
import os

import cv2

from panopticon.config import MODEL_DIRECTORY


def delete_high_brisque_images(directory, brisque_threshold=50):
    """
    Deletes images in a directory with BRISQUE score above a given threshold.

    Args:
        directory (str): The path to the directory containing images.
        brisque_threshold (float): The BRISQUE threshold above which images will be deleted.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    # Paths to BRISQUE model files
    model_path = os.path.join(MODEL_DIRECTORY, "brisque_model_live.yml")
    range_path = os.path.join(MODEL_DIRECTORY, "brisque_range_live.yml")

    if not os.path.exists(model_path) or not os.path.exists(range_path):
        print(f"Error: BRISQUE model files not found at {model_path} or {range_path}")
        return

    # Create BRISQUE object
    brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with BRISQUE score above: {brisque_threshold}")
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure we are working with a file
        if os.path.isfile(file_path):
            try:
                # Load the image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"  - Could not load '{filename}' as an image.")
                    continue

                # Compute the BRISQUE score
                score = brisque.compute(img)

                print(f"  - Processing '{filename}': BRISQUE score = {score[0]:.4f}")

                # If score is above the threshold, delete the file
                if score[0] > brisque_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to high BRISQUE score.")
                    deleted_count += 1

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")

    print(f"Total images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images with high BRISQUE score from a directory.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, default=50, help="The BRISQUE threshold above which images will be deleted (default: 50).")

    args = parser.parse_args()

    delete_high_brisque_images(args.directory, args.threshold)
    print("Script finished.")
