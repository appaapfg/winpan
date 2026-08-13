import argparse
import os

import cv2
import numpy as np


def delete_dark_images(directory, percentage_threshold=50.0, darkness_threshold=50):
    """
    Deletes images in a directory where the percentage of dark pixels is
    above a given threshold.

    Args:
        directory (str): The path to the directory containing images.
        percentage_threshold (float): The percentage of dark pixels above which
                                      images will be deleted.
        darkness_threshold (int): The brightness value (0-255) below which a
                                  pixel is considered dark.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with more than {percentage_threshold}% dark pixels.")
    print(f"Using a darkness threshold of {darkness_threshold}.")
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure we are working with a file
        if os.path.isfile(file_path):
            try:
                # Read the image using OpenCV
                image = cv2.imread(file_path)

                if image is None:
                    print(f"  - Could not read '{filename}'. Skipping.")
                    continue

                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Threshold the image to identify dark pixels.
                # Pixels below darkness_threshold will be white (255), others black (0).
                _, thresholded_image = cv2.threshold(gray_image, darkness_threshold, 255, cv2.THRESH_BINARY_INV)

                # Calculate the percentage of dark pixels
                num_dark_pixels = np.count_nonzero(thresholded_image)
                total_pixels = thresholded_image.size
                percentage_dark_pixels = (num_dark_pixels / total_pixels) * 100

                print(f"  - Processing '{filename}': {percentage_dark_pixels:.2f}% dark pixels")

                # If the percentage of dark pixels is above the threshold, delete the file
                if percentage_dark_pixels > percentage_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to high percentage of dark pixels.")
                    deleted_count += 1

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")

    print(f"Total images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images with a high percentage of dark pixels from a directory.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--percentage", type=float, default=50.0, help="The percentage of dark pixels above which images will be deleted (default: 50.0).")
    parser.add_argument("--darkness", type=int, default=50, help="The brightness value (0-255) below which a pixel is considered dark (default: 50).")

    args = parser.parse_args()

    delete_dark_images(args.directory, args.percentage, args.darkness)
    print("Script finished.")
