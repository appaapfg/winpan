import random
import time

import datasets

# Paths to the datasets
DATASET_PREFIX = "./data_engine/annotated_ds"
DATASET_PREFIX = "./data_engine/coco_ds"
DATASET_A_PATH = f"{DATASET_PREFIX}_A"
DATASET_B_PATH = f"{DATASET_PREFIX}_B"

RANDOM_ACESSES = 100_000


def benchmark_all_annotations(dataset, format_type):
    start_time = time.time()
    total_annotations = 0
    if format_type == "A":
        for item in dataset:
            # Access all lists within the 'objects' dictionary
            if "objects" in item and item["objects"]:
                _ = item["objects"]
                total_annotations += len(item["objects"]["bbox"])
    elif format_type == "B":
        for item in dataset:
            # Iterate through the list of dictionaries
            if "annotations" in item and item["annotations"]:
                _ = item["annotations"]
                total_annotations += len(item["annotations"])
    end_time = time.time()
    print(f"Time to access all annotations in {format_type} format: {end_time - start_time:.4f} seconds for {total_annotations} annotations")


def benchmark_random_annotations(dataset, format_type, num_accesses=RANDOM_ACESSES):
    start_time = time.time()
    accessed_annotations = 0
    if format_type == "A":
        # For format A, we need to pick a random image, then a random annotation within that image
        image_indices = list(range(len(dataset)))
        for _ in range(num_accesses):
            if not image_indices:
                break
            random_image_idx = random.choice(image_indices)
            _ = dataset[random_image_idx]
            accessed_annotations += 1
    elif format_type == "B":
        # For format B, we need to pick a random image, then a random annotation within that image
        # This assumes 'annotations' is a list of dicts per image
        image_indices = list(range(len(dataset)))
        for _ in range(num_accesses):
            if not image_indices:
                break
            random_image_idx = random.choice(image_indices)
            _ = dataset[random_image_idx]
            accessed_annotations += 1
    end_time = time.time()
    print(f"Time to access {accessed_annotations} random annotations in {format_type} format: {end_time - start_time:.4f} seconds")


def main():
    print("Loading datasets...")
    try:
        ds_a = datasets.load_from_disk(DATASET_A_PATH)
        print(f"Dataset A loaded from {DATASET_A_PATH}")
    except Exception as e:
        print(f"Could not load Dataset A: {e}")
        ds_a = None

    try:
        ds_b = datasets.load_from_disk(DATASET_B_PATH)
        print(f"Dataset B loaded from {DATASET_B_PATH}")
    except Exception as e:
        print(f"Could not load Dataset B: {e}")
        ds_b = None

    if ds_a:
        print("\nBenchmarking Dataset A (Format A):")
        benchmark_all_annotations(ds_a, "A")
        benchmark_random_annotations(ds_a, "A")

    if ds_b:
        print("\nBenchmarking Dataset B (Format B):")
        benchmark_all_annotations(ds_b, "B")
        benchmark_random_annotations(ds_b, "B")


if __name__ == "__main__":
    main()
