import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, MobileNetV2ImageProcessor

from panopticon.config import MODEL_DIRECTORY


def delete_night_images(directory, night_threshold=0.5, batch_size=8, device="cuda"):
    """
    Deletes images in a directory classified as nighttime by Woleek/day-night model.

    Args:
        directory (str): The path to the directory containing images.
        night_threshold (float): The threshold above which images will be deleted based on night score (0.0-1.0).
        batch_size (int): Number of images to process in a single batch.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")
    print(f"Deleting images with night score above: {night_threshold}")
    print(f"Batch size: {batch_size}")

    # Load the model and processor
    model_name = "Woleek/day-night"
    print(f"Loading model: {model_name}")
    print(f"Using model directory: {MODEL_DIRECTORY}")

    try:
        processor = MobileNetV2ImageProcessor.from_pretrained(model_name, cache_dir=MODEL_DIRECTORY, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir=MODEL_DIRECTORY, local_files_only=True)
        model.eval()
        model.to(device)
        print("Model loaded successfully.\n")
        print(f"Using device: {device}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Common image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    deleted_count = 0

    # Collect all image files
    image_files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                image_files.append((filename, file_path))

    print(f"Found {len(image_files)} images to process.\n")

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        batch_images = []
        batch_paths = []
        batch_filenames = []

        # Load batch of images
        for filename, file_path in batch_files:
            try:
                image = Image.open(file_path).convert("RGB")
                batch_images.append(image)
                batch_paths.append(file_path)
                batch_filenames.append(filename)
            except Exception as e:
                print(f"  - Could not load '{filename}'. Reason: {e}")

        if not batch_images:
            continue

        try:
            # Process batch
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get night label index
            id2label = model.config.id2label
            night_idx = None

            for idx, label in id2label.items():
                if "night" in label.lower():
                    night_idx = idx
                    break

            if night_idx is None:
                print("  - Warning: Could not find 'night' label in model. Skipping batch.")
                continue

            # Process each image in the batch
            for j, (filename, file_path) in enumerate(zip(batch_filenames, batch_paths)):
                night_score = probabilities[j][night_idx].item()

                print(f"  - Processing '{filename}': Night={night_score:.4f}")

                # If night score is above the threshold, delete the file
                if night_score > night_threshold:
                    os.remove(file_path)
                    print(f"    -> DELETED '{filename}' due to high night score.")
                    deleted_count += 1

        except Exception as e:
            print(f"  - Could not process batch. Reason: {e}")

    print(f"\nTotal images deleted: {deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete nighttime images from a directory using Woleek/day-night classification.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, default=0.5, help="The night score threshold above which images will be deleted (default: 0.5, range: 0.0-1.0).")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process in a single batch (default: 8).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to run inference on (default: cuda).")

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        exit(1)

    # Validate batch size
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
        exit(1)

    delete_night_images(args.directory, args.threshold, args.batch_size, args.device)
    print("Script finished.")
