from __future__ import annotations

import argparse

from pycocotools.coco import COCO

from panopticon.data_engine.util import load_coco


def boxes_are_close(box1, box2, pixel_margin: int) -> bool:
    """
    Compare two xyxy boxes with a given pixel tolerance.
    Returns True if all coordinates differ by <= pixel_margin.
    """
    return all(abs(a - b) <= pixel_margin for a, b in zip(box1, box2))


def group_overlaps(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """
    Given a list of overlapping annotation ID pairs, group them into clusters.
    Example:
        [(12, 14), (12, 15), (14, 16)] -> [[12, 14, 15, 16]]
    """
    if not pairs:
        return []

    # Build adjacency map
    adjacency = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    visited = set()
    groups = []

    # Depth-first search to find connected components
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        group = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            stack.extend(adjacency.get(current, []))
        groups.append(sorted(group))

    return groups


def find_overlapping_bboxes(coco_input: str | COCO | dict, pixel_margin: int = 0):
    """
    Find images with overlapping bounding boxes (class-agnostic).

    Groups all annotations that overlap (directly or indirectly)
    within each image.

    Args:
        coco_input (str | COCO | dict): Path, COCO object, or COCO-format dict.
        pixel_margin (int): Allowed deviation (in pixels) when comparing boxes.
    """
    coco = load_coco(coco_input)
    overlapping_images: dict[int, list[list[int]]] = {}

    for image_id in coco.getImgIds():
        anns = coco.imgToAnns[image_id]
        if not anns:
            continue

        overlaps = []
        boxes = []
        ann_ids = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            xyxy = (round(x), round(y), round(x + w), round(y + h))
            boxes.append(xyxy)
            ann_ids.append(ann["id"])

        # Compare all pairs
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes_are_close(boxes[i], boxes[j], pixel_margin):
                    overlaps.append((ann_ids[i], ann_ids[j]))

        if overlaps:
            groups = group_overlaps(overlaps)
            overlapping_images[image_id] = groups

    # Print results
    if overlapping_images:
        print(f"Images with overlapping bounding boxes (pixel_margin={pixel_margin}):\n")
        for img_id, groups in sorted(overlapping_images.items()):
            print(f"  Image ID {img_id}:")
            for group in groups:
                print(f"    Overlapping annotations: {group}")
            print()
    else:
        print(f"No overlapping bounding boxes found (pixel_margin={pixel_margin}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and group overlapping bounding boxes in a COCO dataset (class-agnostic).")
    parser.add_argument("--input_file", type=str, required=True, help="Path to a COCO JSON file or other valid input for load_coco().")
    parser.add_argument("--pixel_margin", type=int, default=0, help="Tolerance for bounding box equality (0 = exact match, 1 = off-by-one allowed).")

    args = parser.parse_args()
    find_overlapping_bboxes(args.input_file, pixel_margin=args.pixel_margin)
