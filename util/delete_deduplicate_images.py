"""Utility CLI for detecting and deleting duplicate images in a directory."""

import argparse
import os
import subprocess

from imagededup.methods import CNN, DHash, PHash

from panopticon.config import MODEL_DIRECTORY


def delete_duplicates(duplicates, image_dir):
    """Remove duplicate image files from ``image_dir`` based on the provided keys.

    Parameters
    ----------
    duplicates : Iterable[str]
        Collection of filenames relative to ``image_dir`` that should be removed.
    image_dir : str
        Directory that contains the duplicate images.
    """
    for filename in duplicates:
        file_path = os.path.join(image_dir, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                # print(f"Deleted duplicate file: {file_path}")
            else:
                print(f"File not found, skipping: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")


def get_image_files(image_dir):
    """Return a list of full paths to images discovered under ``image_dir``."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    image_files = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, fname))
    return image_files


def get_file_count(directory):
    """Return the number of immediate child files in ``directory``."""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


def _check_czkawka_cli_installed():
    """Return the czkawka CLI command name if available, otherwise ``None``."""
    try:
        subprocess.run(["which", "czkawka_cli"], check=True, capture_output=True)
        return "czkawka_cli"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["which", "czkawka-cli"], check=True, capture_output=True)
            return "czkawka-cli"
        except subprocess.CalledProcessError:
            return None


def czkawka_deduplicate_images(directory, hash_size=16):
    """Run the czkawka CLI repeatedly until duplicate deletions stabilize.

    Parameters
    ----------
    directory : str
        Root directory that will be scanned and cleaned for duplicate images.
    hash_size : int, optional
        Hash size forwarded to czkawka's ``DoubleGradient`` algorithm.
    """
    czkawka_command = _check_czkawka_cli_installed()
    if not czkawka_command:
        print("Error: czkawka_cli or czkawka-cli not found in PATH. Please install it to use this feature.")
        raise SystemExit(1)

    print(f"Processing directory: {directory}")

    previous_count = get_file_count(directory)
    print(f"Initial file count: {previous_count}")

    while True:
        print(f"Running {czkawka_command}...")
        command = [czkawka_command, "image", "--directories", directory, "--hash-alg", "DoubleGradient", "--image-filter", "Lanczos3", "--hash-size", str(hash_size), "--delete-method", "AEB"]
        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=False)

        current_count = get_file_count(directory)
        print(f"Current file count: {current_count}")

        if current_count == previous_count:
            print("File count is stable. Exiting.")
            break

        # Continue deduplication passes until no new files are removed.
        previous_count = current_count


def find_and_delete_duplicates_via_hashing(image_dir, hash_method):
    """Find and delete duplicate images using hashing methods (PHash or DHash)."""
    if hash_method == "phash":
        hasher = PHash()
    elif hash_method == "dhash":
        hasher = DHash()
    else:
        raise ValueError(f"Invalid hash method: {hash_method}")

    # Generate encodings for all images in an image directory
    encodings = hasher.encode_images(image_dir=image_dir)

    # Find duplicates using the generated encodings
    duplicates = hasher.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=5)
    print(f"Found {len(duplicates)} sets of duplicates via hashing.")
    delete_duplicates(duplicates, image_dir)
    print("Duplicate deletion process completed.")


def find_and_delete_duplicates_via_cnn(image_dir, threshold):
    """Find and delete duplicate images using CNN-based embeddings."""
    os.environ["TORCH_HOME"] = MODEL_DIRECTORY
    cnn_encoder = CNN()
    duplicates = cnn_encoder.find_duplicates_to_remove(image_dir=image_dir, min_similarity_threshold=threshold)
    print(f"Found {len(duplicates)} sets of duplicates via embeddings.")
    delete_duplicates(duplicates, image_dir)
    print("Duplicate deletion process completed.")


def main():
    """Parse CLI arguments and coordinate duplicate detection workflows."""
    parser = argparse.ArgumentParser(description="Find and remove duplicate images.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images to deduplicate.")
    parser.add_argument("--hash_method", choices=["phash", "dhash", "none"], default="none", help='Hashing method to use for duplicate detection. Choose "none" to skip hashing.')
    parser.add_argument("--ai_cos_similarity_threshold", type=float, default=0.95, help="Cosine similarity threshold for AI-based duplicate detection. Set to 0 to disable.")
    parser.add_argument("--use_czkawka", action="store_true", help="Use czkawka_cli for duplicate detection.")
    args = parser.parse_args()
    image_dir = args.image_dir

    # Verify directory exists and count image files before processing
    if not os.path.isdir(image_dir):
        print(f"Provided path is not a directory or does not exist: {image_dir}")
        raise SystemExit(1)

    image_files = get_image_files(image_dir)
    print(f"Found {len(image_files)} image file(s) in '{image_dir}'")

    if args.use_czkawka:
        czkawka_deduplicate_images(image_dir)

    if args.hash_method != "none":
        find_and_delete_duplicates_via_hashing(image_dir, args.hash_method)

    if args.ai_cos_similarity_threshold > 0:
        find_and_delete_duplicates_via_cnn(image_dir, args.ai_cos_similarity_threshold)


if __name__ == "__main__":
    main()
