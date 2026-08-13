import argparse
import os

import cv2
import numpy as np
import pyiqa
import torch


def delete_low_clipiqa_images(directory, clipiqa_threshold, device):
    """
    Deletes images in a directory with CLIP-IQA+ score below a given threshold.

    Args:
        directory (str): The path to the directory containing images.
        clipiqa_threshold (float): The CLIP-IQA+ threshold below which images will be deleted.
        device (str): Device to use for computation ('cuda', 'cpu', or 'auto').
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device)

    # Create CLIP-IQA+ metric
    try:
        iqa_metric = pyiqa.create_metric("clipiqa+", device=device)
        print(f"CLIP-IQA+ metric created. Lower better: {iqa_metric.lower_better}")
    except Exception as e:
        print(f"Error: Could not create CLIP-IQA+ metric. Reason: {e}")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with CLIP-IQA+ score below: {clipiqa_threshold}")
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ensure we are working with a file and it's an image
        if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
            try:
                # Load the image to check if it's valid
                img = cv2.imread(file_path)
                if img is None:
                    print(f"  - Could not load '{filename}' as an image.")
                    continue

                # Convert to RGB tensor in [0, 1] format (B, C, H, W)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

                # Compute the CLIP-IQA+ score
                score = iqa_metric(img_tensor)[0]

                print(f"  - Processing '{filename}': CLIP-IQA+ score = {score[0]:.4f}")

                # If score is below the threshold, delete the file (low quality)
                if score[0] < clipiqa_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to low CLIP-IQA+ score.")
                    deleted_count += 1

            except Exception as e:
                print(f"  - Could not process '{filename}'. Reason: {e}")

    print(f"Total images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images with low CLIP-IQA+ score from a directory.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, required=True, help="The CLIP-IQA+ threshold below which images will be deleted.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto' (default).")

    args = parser.parse_args()

    delete_low_clipiqa_images(args.directory, args.threshold, args.device)
    print("Script finished.")
