import argparse
import os

from skimage import io
from skimage.color import rgb2gray
from skimage.restoration import estimate_sigma


def delete_images_by_noise_sigma(directory, sigma_threshold=2.0):
    """
    Deletes images based on the estimated standard deviation (sigma) of the noise.

    This function iterates through images in a directory, estimates the sigma
    value for the image's noise (well-suited for Gaussian noise from camera sensors),
    and deletes the image if its sigma is above the specified threshold.

    Args:
        directory (str): The path to the directory containing images.
        sigma_threshold (float): The noise standard deviation threshold above which
                                 images will be deleted.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with an estimated noise sigma above: {sigma_threshold}")
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            try:
                # Read the image
                image = io.imread(file_path)

                # Convert to grayscale for noise analysis
                if image.ndim == 3:
                    image_gray = rgb2gray(image)
                else:
                    image_gray = image

                # Blindly estimate the standard deviation (sigma) of the noise.
                sigma = estimate_sigma(image_gray, channel_axis=None)

                print(f"  - Processing '{filename}': Estimated Noise Sigma = {sigma:.4f}")

                # If the estimated noise sigma is above the threshold, delete the file
                if sigma > sigma_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to high noise sigma.")
                    deleted_count += 1

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")

    print(f"Total images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images from a directory if their estimated noise standard deviation (sigma) exceeds a threshold.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, default=2.0, help="The noise sigma threshold (default: 2.0). " "Images with a sigma value above this will be deleted.")

    args = parser.parse_args()

    delete_images_by_noise_sigma(args.directory, args.threshold)
    print("Script finished.")
