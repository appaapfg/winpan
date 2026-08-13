"""
COCO Dataset Extensive Statistics Reporter

Usage:
    # Dataset with standard splits train/ valid/ test/
    python coco_dataset_stats.py --dataset-path /path/to/dataset \
            [--save-json stats.json] [--save-cooccurrence cooccur.csv] [--max-cooccurrence-categories 50] \
            [--area-small 32] [--area-medium 96]

    # Single (non-split) dataset containing one _annotations.coco.json in root
    python coco_dataset_stats.py --dataset-path /path/to/single_dataset [--single-split-name all]

Expected dataset directory structure:
 dataset/
  ├── train/
  │   ├── _annotations.coco.json
  │   ├── image1.jpg
  │   └── ...
  ├── valid/
  │   ├── _annotations.coco.json
  │   └── ...
  └── test/
      ├── _annotations.coco.json
      └── ...

Or a non-splitted structure:
 dataset/
  ├── _annotations.coco.json
  ├── image1.jpg
  └── ...

The script prints extensive statistics for each split (if present) and a combined view.
Only standard library modules are used for portability.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ----------------------------- Data Classes ----------------------------- #


@dataclass
class BBox:
    """Represents a bounding box with x, y, width, and height coordinates.

    Attributes:
        x (float): The x-coordinate of the top-left corner.
        y (float): The y-coordinate of the top-left corner.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
    """

    x: float
    y: float
    w: float
    h: float

    def area(self) -> float:
        """Calculates the area of the bounding box.

        Returns:
            float: The area in pixels squared.
        """
        return max(0.0, self.w) * max(0.0, self.h)

    def aspect_ratio(self) -> float:
        """Calculates the aspect ratio (width/height) of the bounding box.

        Returns:
            float: The aspect ratio, or infinity if height is zero.
        """
        if self.h <= 0:
            return float("inf")
        return self.w / self.h

    def clip_within(self, width: int, height: int) -> BBox:
        """Clips the bounding box to fit within the given image dimensions.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            BBox: A new BBox instance clipped to the image bounds.
        """
        x2 = self.x + self.w
        y2 = self.y + self.h
        cx1 = max(0.0, self.x)
        cy1 = max(0.0, self.y)
        cx2 = min(float(width), x2)
        cy2 = min(float(height), y2)
        return BBox(cx1, cy1, max(0.0, cx2 - cx1), max(0.0, cy2 - cy1))

    def iou(self, other: BBox) -> float:
        """Calculates the Intersection over Union (IoU) with another bounding box.

        Args:
            other (BBox): The other bounding box to compare against.

        Returns:
            float: The IoU value between 0.0 and 1.0.
        """
        ax1, ay1, ax2, ay2 = self.x, self.y, self.x + self.w, self.y + self.h
        bx1, by1, bx2, by2 = other.x, other.y, other.x + other.w, other.y + other.h
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


# ----------------------------- Utility Functions ----------------------------- #


def human_int(n: int) -> str:
    """Formats an integer with underscores as thousands separators for readability.

    Args:
        n (int): The integer to format.

    Returns:
        str: The formatted string with underscores.
    """
    return f"{n:,}".replace(",", "_")


def gini_coefficient(counts: Iterable[int]) -> float:
    """Calculates the Gini coefficient for a distribution of counts.

    The Gini coefficient measures inequality in the distribution. A value of 0
    indicates perfect equality, while 1 indicates maximum inequality.

    Args:
        counts (Iterable[int]): An iterable of counts (e.g., category frequencies).

    Returns:
        float: The Gini coefficient between 0.0 and 1.0.
    """
    vals = [c for c in counts if c > 0]
    if not vals:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += i * v
    total = sum(vals)
    return (2 * cum) / (n * total) - (n + 1) / n


def percentile(data: Sequence[float | int], p: float) -> float:
    """Calculates the p-th percentile of a dataset using linear interpolation.

    Args:
        data (Sequence[float | int]): The dataset as a sequence of numbers.
        p (float): The percentile to calculate (0-100).

    Returns:
        float: The p-th percentile value, or NaN if data is empty.
    """
    if not data:
        return float("nan")
    if p <= 0:
        return min(data)
    if p >= 100:
        return max(data)
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[f] * (c - k)
    d1 = data_sorted[c] * (k - f)
    return d0 + d1


# ----------------------------- Core Stats Computation ----------------------------- #


def load_coco_annotations(path: Path) -> dict[str, Any]:
    """Loads COCO annotations from a JSON file.

    Args:
        path (Path): The path to the COCO annotations JSON file.

    Returns:
        dict[str, Any]: The loaded COCO data as a dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class SplitStats:
    """Statistics for a single dataset split (e.g., train, valid, test).

    Attributes:
        split (str): The name of the split.
        images (int): Number of images in the split.
        annotations (int): Number of annotations in the split.
        categories (int): Number of categories in the split.
        images_with_no_annotations (int): Number of images without annotations.
        images_with_annotations (int): Number of images with annotations.
        avg_annotations_per_image (float): Average annotations per image.
        median_annotations_per_image (float): Median annotations per image.
        max_annotations_in_image (int): Maximum annotations in any single image.
        bbox_areas (list[float]): List of bounding box areas.
        aspect_ratios (list[float]): List of bounding box aspect ratios.
        category_annotation_counts (dict[int, int]): Annotation counts per category.
        category_image_counts (dict[int, int]): Image counts per category.
        image_resolution_stats (dict[str, float]): Image resolution statistics.
        bbox_outside_image (int): Number of bounding boxes outside image bounds.
        duplicate_boxes_estimated (int): Estimated number of duplicate boxes.
        segmentation_counts (list[int]): List of segmentation counts per annotation.
        per_category_area_stats (dict[int, dict[str, float]]): Area stats per category.
        cooccurrence (dict[tuple[int, int], int]): Category co-occurrence counts.
    """

    split: str
    images: int
    annotations: int
    categories: int
    images_with_no_annotations: int
    images_with_annotations: int
    avg_annotations_per_image: float
    median_annotations_per_image: float
    max_annotations_in_image: int
    bbox_areas: list[float]
    aspect_ratios: list[float]
    category_annotation_counts: dict[int, int]
    category_image_counts: dict[int, int]
    image_resolution_stats: dict[str, float]
    bbox_outside_image: int
    duplicate_boxes_estimated: int
    segmentation_counts: list[int]
    per_category_area_stats: dict[int, dict[str, float]]
    cooccurrence: dict[tuple[int, int], int]


# Thresholds for duplicate detection (fast heuristic)
IOU_DUPLICATE_THRESHOLD = 0.95
MAX_DUPLICATE_CHECK_PER_IMAGE = 400  # reduce O(N^2)


def compute_split_stats(split: str, coco_input: str | Path | dict, area_small: float, area_medium: float) -> SplitStats:
    """Computes comprehensive statistics for a COCO dataset split.

    This function analyzes images, annotations, categories, bounding boxes,
    aspect ratios, resolutions, and category distributions. It also estimates
    duplicates and computes co-occurrence matrices.

    Args:
        split (str): The name of the dataset split.
        coco_input (str | Path | dict): Path to COCO JSON file, or loaded dict.
        area_small (float): Threshold for small objects (sqrt(area) in pixels).
        area_medium (float): Threshold for medium objects (sqrt(area) in pixels).

    Returns:
        SplitStats: A dataclass containing all computed statistics.

    Raises:
        ValueError: If coco_input is not a valid path, dict, or COCO object.
    """
    if isinstance(coco_input, (str, Path)):
        data = load_coco_annotations(Path(coco_input))
    elif isinstance(coco_input, dict):
        data = coco_input
    elif hasattr(coco_input, "dataset") and isinstance(coco_input.dataset, dict):
        data = coco_input.dataset
    else:
        raise ValueError("coco_input must be a path, dict, or COCO object")
    images = {img["id"]: img for img in data.get("images", [])}
    annotations = data.get("annotations", [])
    categories = {cat["id"]: cat for cat in data.get("categories", [])}

    image_ann_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        image_ann_map[ann["image_id"]].append(ann)

    ann_counts_per_image = [len(v) for v in image_ann_map.values()]
    images_with_no_ann = [img_id for img_id in images if img_id not in image_ann_map]
    images_with_annotations = len(images) - len(images_with_no_ann)

    bbox_areas: list[float] = []
    aspect_ratios: list[float] = []
    segmentation_counts: list[int] = []
    bbox_outside = 0
    duplicate_boxes = 0

    category_annotation_counts: Counter = Counter()
    category_image_set: dict[int, set[int]] = defaultdict(set)
    per_category_areas: dict[int, list[float]] = defaultdict(list)

    # Co-occurrence counting
    cooccurrence: dict[tuple[int, int], int] = defaultdict(int)

    for img_id, anns in image_ann_map.items():
        img_info = images.get(img_id, {})
        iw, ih = img_info.get("width", None), img_info.get("height", None)
        # category presence in this image
        present_cats = set()
        # duplicate detection heuristic: check pairs within image if not too many annotations
        if len(anns) <= MAX_DUPLICATE_CHECK_PER_IMAGE:
            for i in range(len(anns)):
                ai = anns[i]
                bbox_i = BBox(*ai["bbox"])
                for j in range(i + 1, len(anns)):
                    aj = anns[j]
                    if ai.get("category_id") != aj.get("category_id"):
                        continue
                    bbox_j = BBox(*aj["bbox"])
                    if bbox_i.iou(bbox_j) >= IOU_DUPLICATE_THRESHOLD:
                        duplicate_boxes += 1
        for ann in anns:
            cat_raw = ann.get("category_id")
            if cat_raw is None:
                continue  # skip malformed annotation
            try:
                cat = int(cat_raw)
            except Exception:
                continue
            category_annotation_counts[cat] += 1
            present_cats.add(cat)
            bbox = BBox(*ann["bbox"])
            area = ann.get("area", bbox.area())
            bbox_areas.append(area)
            per_category_areas[cat].append(area)
            ar = bbox.aspect_ratio()
            if math.isfinite(ar):
                aspect_ratios.append(ar)
            # segmentation stats
            seg = ann.get("segmentation")
            if isinstance(seg, list):
                segmentation_counts.append(len(seg))
            elif isinstance(seg, dict):  # RLE
                segmentation_counts.append(1)
            # bbox outside image check
            if iw is not None and ih is not None:
                clipped = bbox.clip_within(iw, ih)
                if clipped.w < bbox.w - 1e-6 or clipped.h < bbox.h - 1e-6 or clipped.x > bbox.x + 1e-6 or clipped.y > bbox.y + 1e-6:
                    bbox_outside += 1
        # track categories for image counts
        for c in present_cats:
            category_image_set[c].add(img_id)
        # co-occurrence (upper triangular): count pairs of categories present in this image
        present_list = sorted(present_cats)
        for i in range(len(present_list)):
            for j in range(i, len(present_list)):
                pair = (present_list[i], present_list[j])
                cooccurrence[pair] += 1

    # per-category area stats
    per_category_area_stats: dict[int, dict[str, float]] = {}
    for cat_id, areas in per_category_areas.items():
        per_category_area_stats[cat_id] = {
            "count": len(areas),
            "mean": statistics.fmean(areas) if areas else float("nan"),
            "median": statistics.median(areas) if areas else float("nan"),
            "p10": percentile(areas, 10),
            "p90": percentile(areas, 90),
            "min": min(areas) if areas else float("nan"),
            "max": max(areas) if areas else float("nan"),
            "small_fraction": sum(a < area_small**2 for a in areas) / len(areas) if areas else 0.0,
            "medium_fraction": sum(area_small**2 <= a < area_medium**2 for a in areas) / len(areas) if areas else 0.0,
            "large_fraction": sum(a >= area_medium**2 for a in areas) / len(areas) if areas else 0.0,
        }

    # image resolution stats
    widths = [img.get("width", 0) for img in images.values() if img.get("width")]
    heights = [img.get("height", 0) for img in images.values() if img.get("height")]
    image_resolution_stats = {
        "count": len(widths),
        "width_mean": statistics.fmean(widths) if widths else float("nan"),
        "height_mean": statistics.fmean(heights) if heights else float("nan"),
        "width_median": statistics.median(widths) if widths else float("nan"),
        "height_median": statistics.median(heights) if heights else float("nan"),
        "width_min": min(widths) if widths else float("nan"),
        "width_max": max(widths) if widths else float("nan"),
        "height_min": min(heights) if heights else float("nan"),
        "height_max": max(heights) if heights else float("nan"),
        "megapixels_mean": statistics.fmean([(w * h) / 1_000_000 for w, h in zip(widths, heights)]) if widths else float("nan"),
    }

    return SplitStats(
        split=split,
        images=len(images),
        annotations=len(annotations),
        categories=len(categories),
        images_with_no_annotations=len(images_with_no_ann),
        images_with_annotations=images_with_annotations,
        avg_annotations_per_image=statistics.fmean(ann_counts_per_image) if ann_counts_per_image else 0.0,
        median_annotations_per_image=statistics.median(ann_counts_per_image) if ann_counts_per_image else 0.0,
        max_annotations_in_image=max(ann_counts_per_image) if ann_counts_per_image else 0,
        bbox_areas=bbox_areas,
        aspect_ratios=aspect_ratios,
        category_annotation_counts=dict(category_annotation_counts),
        category_image_counts={k: len(v) for k, v in category_image_set.items()},
        image_resolution_stats=image_resolution_stats,
        bbox_outside_image=bbox_outside,
        duplicate_boxes_estimated=duplicate_boxes,
        segmentation_counts=segmentation_counts,
        per_category_area_stats=per_category_area_stats,
        cooccurrence=dict(cooccurrence),
    )


# ----------------------------- Reporting ----------------------------- #


def summarize_distribution(values: Sequence[float | int]) -> dict[str, float]:
    """Summarizes statistical distribution of a sequence of values.

    Args:
        values (Sequence[float | int]): The values to summarize.

    Returns:
        dict[str, float]: A dictionary with keys 'count', 'mean', 'std',
            'median', 'min', 'p10', 'p90', 'max'.
    """
    if not values:
        return {k: float("nan") for k in ["count", "mean", "std", "median", "min", "p10", "p90", "max"]}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
        "max": max(values),
    }


def print_header(title: str):
    """Prints a formatted header with the given title.

    Args:
        title (str): The title to display in the header.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_split_stats(stats: SplitStats, categories_meta: dict[int, dict[str, Any]]):
    """Prints detailed statistics for a dataset split.

    Args:
        stats (SplitStats): The statistics to print.
        categories_meta (dict[int, dict[str, Any]]): Metadata for categories.
    """
    print_header(f"Split: {stats.split}")
    print(f"Images: {human_int(stats.images)}  Annotations: {human_int(stats.annotations)}  Categories: {human_int(stats.categories)}")
    pct_no_ann = (stats.images_with_no_annotations / stats.images * 100) if stats.images else 0.0
    print(f"Images w/ no annotations: {stats.images_with_no_annotations} ({pct_no_ann:.2f}%)")
    pct_with_ann = (stats.images_with_annotations / stats.images * 100) if stats.images else 0.0
    print(f"Images w/ annotations: {stats.images_with_annotations} ({pct_with_ann:.2f}%)")
    print(f"Avg annotations/image: {stats.avg_annotations_per_image:.2f}  Median: {stats.median_annotations_per_image:.2f}  Max: {stats.max_annotations_in_image}")

    bbox_dist = summarize_distribution(stats.bbox_areas)
    print("BBox area distribution (pixels^2):")
    for k in ["count", "mean", "std", "median", "min", "p10", "p90", "max"]:
        print(f"  {k:>8}: {bbox_dist[k]:.2f}")

    ar_dist = summarize_distribution(stats.aspect_ratios)
    print("Aspect ratio distribution (w/h):")
    for k in ["count", "mean", "std", "median", "min", "p10", "p90", "max"]:
        print(f"  {k:>8}: {ar_dist[k]:.4f}")

    seg_dist = summarize_distribution(stats.segmentation_counts)
    print("Segments per annotation (if polygon lists):")
    for k in ["count", "mean", "std", "median", "min", "p10", "p90", "max"]:
        print(f"  {k:>8}: {seg_dist[k]:.2f}")

    print("Image resolution stats:")
    for k, v in stats.image_resolution_stats.items():
        print(f"  {k:>16}: {v:.3f}")

    print(f"Annotations w/ bbox outside image bounds (partially or fully): {stats.bbox_outside_image}")
    print(f"Estimated duplicate boxes (IoU >= {IOU_DUPLICATE_THRESHOLD} & same category): {stats.duplicate_boxes_estimated}")

    # Category summary
    print("\nPer-category counts (top 25 by annotations):")
    items = sorted(stats.category_annotation_counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
    for cat_id, count in items:
        name = categories_meta.get(cat_id, {}).get("name", str(cat_id))
        img_cnt = stats.category_image_counts.get(cat_id, 0)
        area_info = stats.per_category_area_stats.get(cat_id, {})
        mean_area = area_info.get("mean", float("nan"))
        print(f"  {name:30s} ann: {count:6d} imgs: {img_cnt:5d} mean_area: {mean_area:10.1f}")

    # Imbalance metrics
    counts = list(stats.category_annotation_counts.values())
    if counts:
        gini = gini_coefficient(counts)
        top = max(counts)
        median = statistics.median(counts)
        ratio = top / median if median else float("inf")
        print(f"\nCategory imbalance: Gini={gini:.4f} top/median={ratio:.2f}")

    # Co-occurrence sample
    print("\nCategory co-occurrence (top 20 pairs by count):")
    pairs = sorted(stats.cooccurrence.items(), key=lambda kv: kv[1], reverse=True)[:20]
    for (c1, c2), cnt in pairs:
        n1 = categories_meta.get(c1, {}).get("name", str(c1))
        n2 = categories_meta.get(c2, {}).get("name", str(c2))
        if c1 == c2:
            pair_name = f"{n1} (self)"
        else:
            pair_name = f"{n1} + {n2}"
        print(f"  {pair_name:40s}: {cnt}")


def aggregate_categories(all_meta: list[dict[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    """Aggregates category metadata from multiple sources.

    Args:
        all_meta (list[dict[int, dict[str, Any]]]): List of category metadata dicts.

    Returns:
        dict[int, dict[str, Any]]: Combined category metadata.
    """
    out = {}
    for meta in all_meta:
        out.update(meta)
    return out


# ----------------------------- CLI ----------------------------- #


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv (list[str]): Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description="COCO dataset statistics reporter")
    ap.add_argument("--dataset-path", required=True, help="Path to dataset root containing train/ valid/ test/")
    ap.add_argument("--save-json", help="Optional path to save full stats JSON")
    ap.add_argument("--save-cooccurrence", help="Optional path to save combined co-occurrence matrix CSV")
    ap.add_argument("--max-cooccurrence-categories", type=int, default=50, help="Limit categories for co-occurrence CSV (most frequent)")
    ap.add_argument("--area-small", type=float, default=32.0, help="Small object sqrt(area) threshold (pixels)")
    ap.add_argument("--area-medium", type=float, default=96.0, help="Medium object sqrt(area) threshold (pixels)")
    ap.add_argument("--single-split-name", default="all", help="Name to use for a single (non-split) dataset root")
    return ap.parse_args(argv)


# ----------------------------- Main ----------------------------- #


def main(argv: list[str]) -> int:
    """Main entry point for the COCO dataset statistics reporter.

    Args:
        argv (list[str]): Command-line arguments.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)
    root = Path(args.dataset_path)
    if not root.exists():
        print(f"Dataset path not found: {root}", file=sys.stderr)
        return 1

    start_time = time.time()

    splits = ["train", "valid", "test"]
    split_stats: list[SplitStats] = []
    categories_meta_list: list[dict[int, dict[str, Any]]] = []

    # First attempt standard splits
    for split in splits:
        split_dir = root / split
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        try:
            stats = compute_split_stats(split, ann_file, args.area_small, args.area_medium)
            split_stats.append(stats)
            data = load_coco_annotations(ann_file)
            categories_meta_list.append({c["id"]: c for c in data.get("categories", [])})
        except Exception as e:
            print(f"Error computing stats for split {split}: {e}", file=sys.stderr)

    # If no splits processed, fall back to single root annotation file
    if not split_stats:
        root_ann = root / "_annotations.coco.json"
        if root_ann.exists():
            try:
                stats = compute_split_stats(args.single_split_name, root_ann, args.area_small, args.area_medium)
                split_stats.append(stats)
                data = load_coco_annotations(root_ann)
                categories_meta_list.append({c["id"]: c for c in data.get("categories", [])})
            except Exception as e:
                print(f"Error computing stats for single dataset: {e}", file=sys.stderr)
        else:
            print("No train/valid/test splits found and no root _annotations.coco.json present.")

    categories_meta = aggregate_categories(categories_meta_list)

    if not split_stats:
        print("No splits processed. Exiting.")
        return 1

    print_header("COCO DATASET STATISTICS")
    print(f"Root: {root}")

    for stats in split_stats:
        print_split_stats(stats, categories_meta)

    # Combined stats across splits (simple summations and merges)
    print_header("Combined Category Summary Across Splits")
    combined_cat_ann: Counter = Counter()
    combined_cat_img: Counter = Counter()
    for stats in split_stats:
        combined_cat_ann.update(stats.category_annotation_counts)
        combined_cat_img.update(stats.category_image_counts)

    # Top 30 categories
    for cat_id, count in combined_cat_ann.most_common(30):
        name = categories_meta.get(cat_id, {}).get("name", str(cat_id))
        img_cnt = combined_cat_img.get(cat_id, 0)
        print(f"  {name:30s} ann: {count:6d} imgs: {img_cnt:6d}")

    # Optionally save JSON
    if args.save_json:
        out = {
            "dataset_root": str(root),
            "splits": [
                {
                    "split": s.split,
                    "images": s.images,
                    "annotations": s.annotations,
                    "categories": s.categories,
                    "images_with_no_annotations": s.images_with_no_annotations,
                    "avg_annotations_per_image": s.avg_annotations_per_image,
                    "median_annotations_per_image": s.median_annotations_per_image,
                    "max_annotations_in_image": s.max_annotations_in_image,
                    "bbox_area_stats": summarize_distribution(s.bbox_areas),
                    "aspect_ratio_stats": summarize_distribution(s.aspect_ratios),
                    "segmentation_counts_stats": summarize_distribution(s.segmentation_counts),
                    "image_resolution_stats": s.image_resolution_stats,
                    "bbox_outside_image": s.bbox_outside_image,
                    "duplicate_boxes_estimated": s.duplicate_boxes_estimated,
                    "category_annotation_counts": s.category_annotation_counts,
                    "category_image_counts": s.category_image_counts,
                    "per_category_area_stats": s.per_category_area_stats,
                    "cooccurrence": s.cooccurrence,
                }
                for s in split_stats
            ],
            "categories_meta": categories_meta,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved JSON stats to {args.save_json}")

    # Co-occurrence CSV (combined) for limited categories
    if args.save_cooccurrence:
        # Choose top categories by annotation count
        top_cat_ids = [cat_id for cat_id, _ in combined_cat_ann.most_common(args.max_cooccurrence_categories)]
        # Build combined matrix
        matrix: dict[tuple[int, int], int] = defaultdict(int)
        for s in split_stats:
            for (c1, c2), v in s.cooccurrence.items():
                if c1 in top_cat_ids and c2 in top_cat_ids:
                    matrix[(c1, c2)] += v
        # Symmetrize
        for (c1, c2), v in list(matrix.items()):
            if c1 != c2:
                matrix[(c2, c1)] = v
        # Write CSV
        with open(args.save_cooccurrence, "w", encoding="utf-8") as f:
            header = ["category"] + [str(cid) for cid in top_cat_ids]
            f.write(",".join(header) + "\n")
            for r in top_cat_ids:
                row = [str(r)]
                for c in top_cat_ids:
                    row.append(str(matrix.get((r, c), 0)))
                f.write(",".join(row) + "\n")
        print(f"Saved co-occurrence matrix CSV to {args.save_cooccurrence}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
