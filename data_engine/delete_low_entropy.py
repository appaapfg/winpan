import argparse
import os

from skimage import io, measure
from skimage.color import rgb2gray


def delete_low_entropy_images(directory, entropy_threshold=7.5):
    """
    Deletes images in a directory with entropy below a given threshold.

    Args:
        directory (str): The path to the directory containing images.
        entropy_threshold (float): The entropy threshold below which images will be deleted.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with entropy below: {entropy_threshold}")

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure we are working with a file
        if os.path.isfile(file_path):
            try:
                # Attempt to open the image to verify it's a valid image file
                image = io.imread(file_path)

                # Convert to grayscale if it's a color image
                if image.ndim == 3:
                    image_gray = rgb2gray(image)
                else:
                    image_gray = image

                # Calculate the Shannon entropy
                entropy = measure.shannon_entropy(image_gray)

                print(f"  - Processing '{filename}': Entropy = {entropy:.4f}")

                # If entropy is below the threshold, delete the file
                if entropy < entropy_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to low entropy.")

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images with low entropy from a directory.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, default=7.5, help="The entropy threshold below which images will be deleted (default: 0.25).")

    args = parser.parse_args()

    delete_low_entropy_images(args.directory, args.threshold)
    print("Script finished.")
