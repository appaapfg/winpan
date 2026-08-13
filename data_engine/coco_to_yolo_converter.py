import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from ultralytics.data.converter import convert_coco

"""
Convert COCO dataset to YOLO format.

This script converts a COCO-formatted dataset into Ultralytics YOLO format.
It processes train, valid, and test splits, generates YOLO annotations, and creates a unified data.yaml configuration file.

Dataset format:
```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

Args:
    --coco-dir (str): Path to the input COCO dataset directory. Defaults to "./con-coco/".
    --yolo-dir (str): Path to the output YOLO dataset directory. Defaults to "./ttt1/".
    --use-symlinks (bool): If specified, creates symlinks for images instead of copying them. Defaults to False (copying).
    --use-segments (bool): If specified, use segments in conversion. Defaults to False.

Examples:
    Convert a COCO dataset to YOLO format by copying images:
    python coco_to_yolo_converter.py --coco-dir /path/to/coco --yolo-dir /path/to/yolo

    Convert a COCO dataset to YOLO format using symlinks:
    python coco_to_yolo_converter.py --coco-dir /path/to/coco --yolo-dir /path/to/yolo --use-symlinks

Notes:
    - Requires ultralytics library for conversion.
    - By default, images are copied to the output directory. Use `--use-symlinks` to create symlinks instead.
    - Handles nested labels directory issues from some converter versions.
"""

SPLITS = ["test", "valid", "train"]


def coco_json_to_data_yaml(coco_json_file_path, output_data_yaml_file_path):
    """Create a Ultralytics YOLO `data.yaml` file from a COCO annotations JSON.

    This helper reads the `categories` section of a COCO annotations file (the
    `_annotations.coco.json` produced by many labeling tools) and writes a
    `data.yaml` compatible with Ultralytics YOLO training / validation routines.

    The resulting YAML contains:
      - `path`: Root path (relative in this script) to the converted dataset directory
      - `train`, `val`, `test`: Subdirectory paths (relative to `path`) pointing to the image folders
      - `names`: Mapping from category id (int) to human‑readable class name

    Notes
    -----
    * The function does not validate that the train/val/test folders actually
      exist—`convert_coco` (from `ultralytics.data.converter`) is typically run
      beforehand to create the YOLO directory structure.
    * The YAML is overwritten if it already exists.

    Parameters
    ----------
    coco_json_file_path : str | PathLike
        Path to a COCO annotations JSON containing a top-level `categories` list.
    output_data_yaml_file_path : str | PathLike
        Destination path for the generated `data.yaml` file.

    Raises
    ------
    FileNotFoundError
        If `coco_json_file_path` does not exist.
    KeyError
        If the JSON does not contain a `categories` key.
    json.JSONDecodeError
        If the file is not valid JSON.

    Returns
    -------
    None

    Example
    -------
    >>> from ultralytics.data.converter import coco_json_to_data_yaml, convert_coco
    >>> # Assume a COCO dataset under ./my_coco_dataset with train/valid/test splits
    >>> COCO_DIR = './my_coco_dataset'
    >>> YOLO_DIR = './my_ultralytics_dataset'
    >>> for split in ['train', 'valid', 'test']:
    ...     convert_coco(f"{COCO_DIR}/{split}", f"{YOLO_DIR}/{split}", use_segments=False)
    ...
    >>> # After conversion, create (or overwrite) the unified data.yaml
    >>> coco_json_to_data_yaml(f"{COCO_DIR}/train/_annotations.coco.json", f"{YOLO_DIR}/data.yaml")
    >>> # You can now train: `yolo detect train data={YOLO_DIR}/data.yaml model=yolo12n.pt`
    """
    coco_categories = json.load(open(coco_json_file_path))["categories"]
    data = {"path": ULTRALYTICS_DATASET_DIRECTORY, "train": "train/images", "val": "valid/images", "test": "test/images", "names": {coco_category["id"]: coco_category["name"] for coco_category in coco_categories}}

    with open(output_data_yaml_file_path, "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("--coco-dir", default="./con-coco/", help="Path to COCO dataset directory")
    parser.add_argument("--yolo-dir", default="./ttt1/", help="Path to output YOLO dataset directory")
    parser.add_argument("--use-symlinks", action="store_true", help="Use symlinks instead of copying images.")
    parser.add_argument("--use-segments", action="store_true", help="Use segments in conversion.")
    args = parser.parse_args()

    COCO_DATASET_DIRECTORY = args.coco_dir
    ULTRALYTICS_DATASET_DIRECTORY = args.yolo_dir

    for split in SPLITS:
        convert_coco(
            f"{COCO_DATASET_DIRECTORY}/{split}",
            f"{ULTRALYTICS_DATASET_DIRECTORY}/{split}",
            use_segments=args.use_segments,
            cls91to80=False,
        )

        # Flatten wrongly nested labels directory that some converter versions create:
        #   {split}/labels/_annotations.coco/*.txt  -->  {split}/labels/*.txt
        # This keeps existing files (if any) and overwrites on name collision.
        out_split_dir = Path(ULTRALYTICS_DATASET_DIRECTORY) / split
        labels_dir = out_split_dir / "labels"
        nested_labels_dir = labels_dir / "_annotations.coco"
        if nested_labels_dir.is_dir():
            try:
                moved_any = False
                for p in nested_labels_dir.glob("*.txt"):
                    target = labels_dir / p.name
                    try:
                        if target.exists():
                            target.unlink()
                        p.rename(target)
                        moved_any = True
                    except Exception as move_err:
                        print(f"[WARN] Could not move label file '{p}': {move_err}")
                # Remove the now-empty nested directory tree if possible
                try:
                    if not any(nested_labels_dir.iterdir()):
                        nested_labels_dir.rmdir()
                    # If labels/ becomes empty aside from prior nested dir, leave it (harmless)
                except Exception as cleanup_err:
                    print(f"[WARN] Cleanup of nested labels directory failed: {cleanup_err}")
                if moved_any:
                    print(f"[INFO] Flattened nested labels for split '{split}'")
            except Exception as e:
                print(f"[WARN] Failed to flatten labels for split '{split}': {e}")

        # Replace the generated images directory with a symlink to the original split (or its images subdir if present)
        images_dir = out_split_dir / "images"
        orig_split_dir = Path(COCO_DATASET_DIRECTORY) / split
        candidate_images_subdir = orig_split_dir / "images"
        orig_images_dir = candidate_images_subdir if candidate_images_subdir.is_dir() else orig_split_dir

        try:
            if images_dir.exists() or images_dir.is_symlink():
                if images_dir.is_dir() and not images_dir.is_symlink():
                    shutil.rmtree(images_dir)
                else:
                    images_dir.unlink()

            if args.use_symlinks:
                os.symlink(os.path.relpath(orig_images_dir, out_split_dir), images_dir)
            else:
                shutil.copytree(orig_images_dir, images_dir)
        except Exception as e:
            print(f"[WARN] Failed to create images symlink for split '{split}': {e}")

        coco_json_to_data_yaml(f"{COCO_DATASET_DIRECTORY}/{split}/_annotations.coco.json", f"{ULTRALYTICS_DATASET_DIRECTORY}/data.yaml")
