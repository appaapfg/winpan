"""
Module for visualizing COCO dataset images with annotations.

This module provides functionality to load COCO annotations, select images, and display them with drawn annotations using matplotlib and PIL.

Examples:
    python visualize.py --annotation data/_annotations.coco.json --images data/images/
"""

import argparse
import os
import random
import sys
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

# Ensure repo root is on sys.path so we can import draw.py when running this file directly
_FILE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(_FILE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# draw is imported lazily inside visualize() after path setup


# Default paths (relative to repo root)
ANNOTATION_FILE = "data_engine/keyframes_peter_pan/_annotations.coco.json"
IMAGE_DIRECTORY = "data_engine/keyframes_peter_pan/"


def _ensure_image_path(base_dir: str, file_name: str) -> str:
    """Return an existing absolute image path, trying common layouts.

    Tries: base_dir/file_name, base_dir/images/file_name
    """
    direct = os.path.join(base_dir, file_name)
    if os.path.exists(direct):
        return direct
    nested = os.path.join(base_dir, "images", file_name)
    if os.path.exists(nested):
        return nested
    # As-is – let caller handle error
    return direct


def _load_categories(coco: COCO) -> list[Any]:
    cat_ids = coco.getCatIds()
    return coco.loadCats(cat_ids)  # type: ignore[return-value]


def _pick_image(coco: COCO, index: int | None) -> Any:
    img_ids = coco.getImgIds()
    if not img_ids:
        raise RuntimeError("No images found in COCO annotations.")
    if index is None:
        img_id = random.choice(img_ids)
    else:
        # Make index stable across runs by sorting IDs
        img_ids_sorted = sorted(img_ids)
        img_id = img_ids_sorted[index % len(img_ids_sorted)]
    imgs = coco.loadImgs([img_id])
    if not imgs:
        raise RuntimeError(f"Failed to load image metadata for id={img_id}")
    return imgs[0]


def _load_annotations_for_image(coco: COCO, image_id: int) -> list[Any]:
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    return coco.loadAnns(ann_ids)  # type: ignore[return-value]


def visualize(
    annotation_file: str = ANNOTATION_FILE,
    image_directory: str = IMAGE_DIRECTORY,
    index: int | None = None,
    image_id: int | None = None,
    file_name: str | None = None,
    out_path: str | None = None,
    show: bool = True,
) -> tuple[str, str | None]:
    """Visualize a dataset image with annotations using build_annotated_image.

    Returns (title, saved_path_or_None).
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")

    coco = COCO(annotation_file)
    categories = _load_categories(coco)

    img_info: Any | None = None
    if file_name is not None:
        # Find by file_name
        img_ids = coco.getImgIds()
        candidates = coco.loadImgs(img_ids)
        for info in candidates:
            if info.get("file_name") == file_name:
                img_info = info
                break
        if img_info is None:
            raise ValueError(f"file_name not found in annotations: {file_name}")
    elif image_id is not None:
        imgs = coco.loadImgs([image_id])
        if not imgs:
            raise ValueError(f"image_id not found in annotations: {image_id}")
        img_info = imgs[0]
    else:
        img_info = _pick_image(coco, index)

    assert img_info is not None
    file_name = str(img_info["file_name"]) if img_info.get("file_name") is not None else None
    if not file_name:
        raise ValueError("Image metadata missing file_name")
    img_path = _ensure_image_path(image_directory, file_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Load image (RGB) and annotations
    image = Image.open(img_path).convert("RGB")
    anns = _load_annotations_for_image(coco, img_info["id"]) or []

    # Build annotated RGB numpy array
    from panopticon.draw import (
        draw_annotations_on_image,  # local import to avoid module-level import order issue
    )

    annotated = draw_annotations_on_image(image, anns, categories)

    # Display or save
    title = f"{file_name}  (id={img_info['id']}, anns={len(anns)})"
    saved_path: str | None = None
    if out_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.imsave(out_path, annotated)
        saved_path = out_path
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return title, saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a dataset image with annotations.")
    parser.add_argument("--annotation", default=ANNOTATION_FILE, help="Path to COCO annotations JSON file.")
    parser.add_argument("--images", default=IMAGE_DIRECTORY, help="Directory containing dataset images.")
    parser.add_argument("--index", type=int, default=None, help="Index of image (stable across runs).")
    parser.add_argument("--image-id", type=int, default=None, help="COCO image id to visualize.")
    parser.add_argument("--file", dest="file_name", default=None, help="file_name to visualize (as in COCO JSON).")
    parser.add_argument("--out", dest="out_path", default=None, help="Optional path to save the visualization.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the window (useful in headless).")
    args = parser.parse_args()

    title, saved = visualize(
        annotation_file=args.annotation,
        image_directory=args.images,
        index=args.index,
        image_id=args.image_id,
        file_name=args.file_name,
        out_path=args.out_path,
        show=not args.no_show,
    )
    print(title)
    if saved:
        print(f"Saved: {saved}")
