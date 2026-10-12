# from ultralytics.data.converter import convert_coco
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

# Convert COCO dataset to YOLO format
"""
Dataset format
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


def create_yolo_dataset(coco_dataset_directory, yolo_dataset_directory, copy_images=False, use_segments=True):
    for split in SPLITS:
        convert_coco(
            f"{coco_dataset_directory}/{split}",
            f"{yolo_dataset_directory}/{split}",
            use_segments=use_segments,
            cls91to80=False,
        )

        # Flatten wrongly nested labels directory that some converter versions create:
        #   {split}/labels/_annotations.coco/*.txt  -->  {split}/labels/*.txt
        # This keeps existing files (if any) and overwrites on name collision.
        out_split_dir = Path(yolo_dataset_directory) / split
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

        # Replace the generated images directory with symlinks to individual images from the original split
        images_dir = out_split_dir / "images"
        orig_split_dir = Path(coco_dataset_directory) / split
        candidate_images_subdir = orig_split_dir / "images"
        orig_images_dir = candidate_images_subdir if candidate_images_subdir.is_dir() else orig_split_dir

        try:
            # Ensure images_dir is a directory
            if images_dir.exists():
                if images_dir.is_symlink():
                    images_dir.unlink()
                elif images_dir.is_dir():
                    shutil.rmtree(images_dir)
            images_dir.mkdir(parents=True, exist_ok=True)

            # Symlink individual image files
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
            for ext in image_extensions:
                for img_path in orig_images_dir.glob(ext):
                    symlink_path = images_dir / img_path.name
                    if symlink_path.exists():
                        symlink_path.unlink()
                    if copy_images:
                        shutil.copy2(img_path, symlink_path)
                    else:
                        os.symlink(os.path.relpath(img_path, images_dir), symlink_path)
        except Exception as e:
            print(f"[WARN] Failed to create image symlinks for split '{split}': {e}")

        coco_json_to_data_yaml(
            f"{coco_dataset_directory}/{split}/_annotations.coco.json",
            f"{yolo_dataset_directory}/data.yaml",
        )


# copied from ultalyticss at 3-10-2025
def convert_coco(
    labels_dir: str = "../coco/annotations/",
    save_dir: str = "coco_converted/",
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = False,
    lvis: bool = False,
):
    """
    Convert COCO dataset annotations to a YOLO annotation format suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        lvis (bool, optional): Whether to convert data in lvis dataset way.

    Examples:
        >>> from ultralytics.data.converter import convert_coco

        Convert COCO annotations to YOLO format
        >>> convert_coco("coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)

        Convert LVIS annotations to YOLO format
        >>> convert_coco("lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
    """
    # ADDED IMPORTS
    from collections import defaultdict

    import numpy as np
    from ultralytics.data.converter import coco91_to_coco80_class, merge_multi_segment
    from ultralytics.utils import LOGGER  # , TQDM
    from ultralytics.utils.files import increment_path
    from ultralytics.utils.tqdm import TQDM

    # Create dataset directory
    save_dir = increment_path(save_dir)  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Convert classes
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            # NOTE: create folders for both train and val in advance,
            # since LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Create image dict
        images = {f"{x['id']:d}": x for x in data["images"]}
        # Create image-annotations dict
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        image_txt = []
        # Write labels file
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                # if ann.get("iscrowd", False):  # DISABLED iscrowd checks
                #     continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"]  # class MODIFIED removed the - 1
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist())

            # Write
            with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (*(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if lvis:
            filename = Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("--coco-dir", help="Path to COCO dataset directory")
    parser.add_argument("--yolo-dir", help="Path to output YOLO dataset directory")
    parser.add_argument("--use-segments", action=argparse.BooleanOptionalAction, default=True, help="Include segmentation masks")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=False, help="Copy images instead of creating symlinks")
    args = parser.parse_args()

    COCO_DATASET_DIRECTORY = args.coco_dir
    ULTRALYTICS_DATASET_DIRECTORY = args.yolo_dir

    create_yolo_dataset(COCO_DATASET_DIRECTORY, ULTRALYTICS_DATASET_DIRECTORY, copy_images=args.copy_images, use_segments=args.use_segments)
