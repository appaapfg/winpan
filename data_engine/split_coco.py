"""
Module for splitting COCO datasets into train, validation, and test sets.

This script provides functionality to load a COCO dataset, split it into subsets, and save them in COCO format for machine learning tasks.

Examples:
    python split_coco.py --image_dir /path/to/images --annotations_path /path/to/annotations.json --output_dir /path/to/output
"""

import argparse
import os

import supervision as sv


def split_and_save_coco(image_dir, annotations_path, output_dir, train_split, validation_split, random_state, shuffle):
    """
    Loads a COCO dataset, splits it into train, validation, and test sets, and saves them.

    Args:
        image_dir (str): Path to the directory with original images.
        annotations_path (str): Path to the original COCO annotations JSON file.
        output_dir (str): Directory to save the split datasets.
        train_split (float): Ratio for the training set split.
        validation_split (float): Ratio of the remaining data to use for validation.
        random_state (int): Random state for shuffling.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tuple: A tuple containing the sizes of the original dataset, training set,
               validation set, and test set (original_size, train_size, valid_size, test_size).
    """
    # 1. Load the COCO dataset
    dataset = sv.DetectionDataset.from_coco(images_directory_path=image_dir, annotations_path=annotations_path)

    # 2. Split the dataset into training and a temporary set (validation + test)
    train_dataset, temp_dataset = dataset.split(split_ratio=train_split, random_state=random_state, shuffle=shuffle)

    # 3. Split the temporary set into validation and test sets
    valid_dataset, test_dataset = temp_dataset.split(split_ratio=validation_split, random_state=random_state, shuffle=shuffle)

    # 4. Save the new datasets into COCO format
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_dataset.as_coco(annotations_path=os.path.join(train_dir, "annotations.coco.json"), images_directory_path=train_dir)

    valid_dataset.as_coco(annotations_path=os.path.join(valid_dir, "annotations.coco.json"), images_directory_path=valid_dir)

    test_dataset.as_coco(annotations_path=os.path.join(test_dir, "annotations.coco.json"), images_directory_path=test_dir)

    return len(dataset), len(train_dataset), len(valid_dataset), len(test_dataset)


def main():
    parser = argparse.ArgumentParser(description="Split a COCO dataset into training, validation, and test sets.")
    parser.add_argument("--image_dir", required=True, help="Path to the directory with original images.")
    parser.add_argument("--annotations_path", required=True, help="Path to the original COCO annotations JSON file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the split datasets.")
    parser.add_argument("--train_split", type=float, default=0.7, help="Ratio for the training set split (default: 0.7).")
    parser.add_argument("--validation_split", type=float, default=0.5, help="Ratio of the remaining data to use for validation (default: 0.5).")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for shuffling (default: 42).")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling of the dataset.")

    args = parser.parse_args()

    original_size, train_size, valid_size, test_size = split_and_save_coco(image_dir=args.image_dir, annotations_path=args.annotations_path, output_dir=args.output_dir, train_split=args.train_split, validation_split=args.validation_split, random_state=args.random_state, shuffle=not args.no_shuffle)

    print(f"Original dataset size: {original_size}")
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {valid_size}")
    print(f"Test dataset size: {test_size}")


if __name__ == "__main__":
    main()
