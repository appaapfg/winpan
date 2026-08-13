import argparse
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from panopticon.my_inference.detect import Detector

# TODO load images in the main loop

OBJECT_CLASSES = [
    {"supercategory": "person", "id": 1, "name": "person"},
    {"supercategory": "animal", "id": 2, "name": "dog"},
    {"supercategory": "vehicle", "id": 3, "name": "ship"},
]
OBJECT_CLASSES = [
    {"supercategory": "other", "id": 10, "name": "tree"},
]


def convert_tensors_to_lists(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(elem) for elem in obj]
    else:
        return obj


def annotate_coco_dataset(input_file: str | COCO, output_file: str | None, model_type, threshold, batch_size, inference_resolution: tuple[int, int], object_categories=OBJECT_CLASSES):
    LABEL2ID = {cls["name"]: cls["id"] for cls in object_categories}
    # ID2LABEL = {cls["id"]: cls["name"] for cls in object_categories}

    # Initialize COCO api for instance annotations
    if isinstance(input_file, str):
        coco = COCO(input_file)
    else:
        coco = input_file
    assert isinstance(coco, COCO), "Input must be a COCO object or a path to a COCO JSON file."

    TEXT_LABELS = [cls["name"] for cls in object_categories]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLD = threshold
    BATCH_SIZE = batch_size
    MODEL_TYPE = model_type
    detector = Detector(model_type=MODEL_TYPE, device=DEVICE)

    print(f"Successfully loaded COCO dataset from {input_file}")

    # Get all image IDs
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)

    # Prepare images for detection
    image_paths = [os.path.join(os.path.dirname(input_file), img["file_name"]) for img in images]
    pil_images = []
    for img_path in tqdm(image_paths, desc="Loading images"):  # Wrap with tqdm
        try:
            pil_images.append(Image.open(img_path).convert("RGB"))
        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}. Skipping.")
            pil_images.append(None)  # Append None to maintain batch size, will filter later

    # Filter out None values (for images not found)
    valid_images = [img for img in pil_images if img is not None]
    valid_img_ids = [img_ids[i] for i, img in enumerate(pil_images) if img is not None]

    if not valid_images:
        print("No valid images found to annotate.")
        return

    # Perform detection in batches
    new_annotations = []
    for i in tqdm(range(0, len(valid_images), BATCH_SIZE), desc="Detecting objects"):  # Wrap with tqdm
        batch_images = valid_images[i : i + BATCH_SIZE]
        batch_img_ids = valid_img_ids[i : i + BATCH_SIZE]

        results = detector.detect(batch_images, [TEXT_LABELS] * len(batch_images), threshold=THRESHOLD, size=inference_resolution)

        for j, result in enumerate(results):
            img_id = batch_img_ids[j]
            image_info = coco.loadImgs(img_id)[0]
            img_width = image_info["width"]
            img_height = image_info["height"]

            boxes = result["boxes"]
            boxes = torchvision.ops.clip_boxes_to_image(boxes, (img_height, img_width))
            labels = result["text_labels"]
            scores = result["scores"]
            masks = result.get("masks", None)  # Optional segmentation masks (Tensor[N,H,W])

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            areas = areas.tolist()
            coco_boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
            bbox_list = coco_boxes.tolist()
            category_list = labels
            category_id_list = [LABEL2ID[label] for label in labels]
            confidence_list = scores.tolist()

            for k in range(len(boxes)):
                ann = {
                    "id": len(coco.anns) + len(new_annotations) + 1,  # Unique ID for new annotation
                    "image_id": img_id,
                    "category_id": category_id_list[k],
                    "category_str": category_list[k],
                    "area": areas[k],
                    "bbox": bbox_list[k],
                    "iscrowd": 1,
                    "bbox_confidence": confidence_list[k],
                }
                # If masks are available, add COCO RLE segmentation
                if masks is not None:
                    mask_k = masks[k]
                    assert isinstance(mask_k, torch.Tensor)

                    mask_arr = mask_k.detach().cpu().numpy()
                    # Binarize (assume probabilities if not already 0/1)
                    assert mask_arr.dtype == np.uint8
                    assert mask_arr.ndim == 2
                    rle = mask_utils.encode(np.asfortranarray(mask_arr))
                    # pycocotools outputs bytes for counts; convert to utf-8 str for json
                    counts_val = rle.get("counts")
                    if isinstance(counts_val, bytes):
                        rle["counts"] = counts_val.decode("utf-8")
                    ann["segmentation"] = rle
                new_annotations.append(ann)

    # Add new annotations to the COCO object
    coco.dataset["annotations"].extend(new_annotations)
    # Add object classes to the 'categories' field before converting
    coco.dataset["categories"].extend(object_categories)
    # Convert any remaining Tensors in coco.dataset to lists before saving
    final_dataset = convert_tensors_to_lists(coco.dataset)

    # Save the updated COCO dataset to a new JSON file
    if output_file is not None and output_file.lower() != "none":
        with open(output_file, "w") as f:
            json.dump(final_dataset, f)

    print(f"Successfully annotated and saved COCO dataset to {output_file}")

    return coco


def main():
    parser = argparse.ArgumentParser(description="Annotate COCO dataset with object detections.")
    parser.add_argument("--input_file", type=str, default="./data_engine/keyframes_peter_pan/_unlabeled.coco.json", help="Path to the input COCO annotation file.")
    parser.add_argument("--output_file", type=str, default="None", help="Path to save the annotated COCO dataset.")
    parser.add_argument("--model_type", type=str, default="yoloe_11s", help="Type of detection model to use (e.g., 'grounding_dino_base', 'yoloe_11s').")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection confidence threshold.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for object detection.")
    parser.add_argument("--inference_resolution", type=int, nargs=2, default=[640, 640], help="Inference resolution (height, width).")

    args = parser.parse_args()

    annotate_coco_dataset(args.input_file, args.output_file, args.model_type, args.threshold, args.batch_size, inference_resolution=tuple(args.inference_resolution))


if __name__ == "__main__":
    main()
