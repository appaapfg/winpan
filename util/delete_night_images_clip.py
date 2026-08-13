import argparse
import os

# Ignore future warnings for cleaner output
import warnings

import open_clip
import torch
from PIL import Image
from timm.utils.model import reparameterize_model
from tqdm import tqdm

from panopticon.config import MODEL_DIRECTORY

warnings.filterwarnings("ignore", category=FutureWarning)

# Device setup: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model_and_preprocess(model_name: str, pretrained_path: str):
    """Load the model and its preprocessing pipeline."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_path)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def delete_images_clip_based(directory, threshold, model_name, inference_resolution=(256, 256), batch_size=8, accept_label="A picture taken during the day", delete_label="A picture taken during the night"):
    """Delete images classified as 'night photos' using batch inference."""
    # Model and tokenizer setup
    model, preprocess, tokenizer = load_model_and_preprocess(model_name, f'{MODEL_DIRECTORY}/{model_name.lower().replace("-", "_")}.pt')

    # Reparameterize model for inference on CPU to avoid CUDNN errors
    model.eval()
    model = reparameterize_model(model)
    model.to(device)

    # Encode the text features once (no need to repeat for each image)
    text_tensor = tokenizer([accept_label, delete_label]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tensor)
        text_features /= text_features.norm(dim=-1, keepdim=True) + 1e-8

    # Collect all image paths
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Process in batches
    deleted_count = 0
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB").resize(inference_resolution)
                batch_tensors.append(preprocess(image))
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not batch_tensors:
            continue

        # Stack into a single tensor and send to device
        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True) + 1e-8
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Loop through batch results
        for j, path in enumerate(batch_paths):
            delete_prob = text_probs[j][1].item()
            print(f"Image {os.path.basename(path)} has a delete probability of {delete_prob:.4f}")

            if delete_prob > threshold:
                try:
                    os.remove(path)
                    print(f"Deleted {os.path.basename(path)} (classified for deletion).")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")

    print(f"Total images deleted: {deleted_count}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Delete images classified as 'night photos' from a directory.")
    parser.add_argument("directory", help="The path to the directory containing images.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classifying an image as night (0.0 to 1.0). Default: 0.5")
    parser.add_argument("--model_name", type=str, default="MobileCLIP2-S0", help="Name of the CLIP model to use. Default: MobileCLIP2-S0")
    parser.add_argument("--inference_resolution", type=int, nargs=2, default=[256, 256], help="Resolution to resize images for inference (width height). Default: [256, 256]")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to process in each batch. Default: 1")
    args = parser.parse_args()

    # Delete dark images classified as night photos
    delete_images_clip_based(args.directory, args.threshold, args.model_name, inference_resolution=tuple(args.inference_resolution), batch_size=args.batch_size)

    print("Script finished.")


# Main execution
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    main()
