import argparse
import os

import cv2


def delete_low_laplacian_images(directory, laplacian_threshold=100.0):
    """
    Deletes images in a directory with Laplacian variance below a given threshold.

    Args:
        directory (str): The path to the directory containing images.
        laplacian_threshold (float): The Laplacian variance threshold below which images will be deleted.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with Laplacian variance below: {laplacian_threshold}")
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure we are working with a file
        if os.path.isfile(file_path):
            try:
                # Attempt to open the image to verify it's a valid image file
                image = cv2.imread(file_path)

                if image is None:
                    print(f"  - Could not read '{filename}' as an image. Skipping.")
                    continue

                # Convert to grayscale if it's a color image
                if len(image.shape) == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image_gray = image

                # Calculate the Laplacian variance
                laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
                variance = laplacian.var()

                print(f"  - Processing '{filename}': Laplacian variance = {variance:.4f}")

                # If Laplacian variance is below the threshold, delete the file
                if variance < laplacian_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to low Laplacian variance.")
                    deleted_count += 1

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")

    print(f"Total images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images with low Laplacian variance from a directory.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, default=100.0, help="The Laplacian variance threshold below which images will be deleted (default: 100.0).")

    args = parser.parse_args()

    delete_low_laplacian_images(args.directory, args.threshold)
    print("Script finished.")
