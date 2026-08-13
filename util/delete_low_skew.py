import argparse
import os

import cv2
import numpy as np


def delete_dark_images_by_peak(directory, peak_threshold=50):
    """
    Deletes images in a directory if their histogram's peak is below a given threshold.

    A dark image will have a histogram with a peak (highest frequency of pixels)
    at a low intensity value (closer to black).

    Args:
        directory (str): The path to the directory containing images.
        peak_threshold (int): The intensity value threshold for the histogram peak.
                               Images with a peak below this value will be deleted.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with a histogram peak below: {peak_threshold}")
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure we are working with a file
        if os.path.isfile(file_path):
            try:
                # Read the image in grayscale using OpenCV
                image_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Check if the image was loaded successfully
                if image_gray is None:
                    print(f"  - Could not read '{filename}'. Skipping.")
                    continue

                # Calculate the histogram.
                # The arguments are: [image], [channel], mask, [histsize], [ranges]
                hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

                # Find the peak of the histogram (the bin with the most pixels)
                peak_location = np.argmax(hist)

                print(f"  - Processing '{filename}': Histogram Peak at intensity = {peak_location}")

                # If the peak is below the threshold, delete the file
                if peak_location < peak_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to dark histogram peak.")
                    deleted_count += 1

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")

    print(f"Total images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete dark images from a directory based on histogram peak location.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=int, default=50, help="The histogram peak intensity threshold (0-255). Images with a peak below this will be deleted (default: 50).")

    args = parser.parse_args()

    delete_dark_images_by_peak(args.directory, args.threshold)
    print("Script finished.")
