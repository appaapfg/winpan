import copy
import hashlib

import numpy as np
import torch
from transformers import (
    Sam3Config,
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)
from ultralytics import SAM as UltralyticsSAM
from ultralytics.models.sam import Predictor, SAM2Predictor

from panopticon.config import MODEL_DIRECTORY

# Model configs for segmenter
MODEL_CONFIGS = {
    "mobile_sam": "mobile_sam.pt",
    "sam2.1_tiny": "sam2.1_t.pt",
    "sam2.1_small": "sam2.1_s.pt",
    "sam2.1_base": "sam2.1_b.pt",
    "sam2.1_large": "sam2.1_l.pt",
    "sam3_pcs": "facebook/sam3",  # PCS
    "sam3_pvs": "facebook/sam3",  # PVS
}


class Segmenter:
    SUPPORTED_MODELS = list(MODEL_CONFIGS.keys())

    def __init__(self, model_type: str | None, device: str = "cpu"):
        """Initialize a Segmenter.

        Args:
            model_type: One of SUPPORTED_MODELS or None. If None, a placeholder Segmenter
                        is created with no underlying model (mirrors Detector handling of None).
            device: Torch device string.
        """
        self.DEVICE = device
        self.last_image_hash = None
        self.text_embedding_cache = {}

        # Handle uninitialized / deferred loading case
        if model_type is None:
            self.model = None
            self.model_type = None
            if self.DEVICE == "cuda":
                torch.cuda.empty_cache()
            return

        self.model_type = model_type
        model_name = MODEL_CONFIGS.get(model_type)
        if not model_name:
            raise ValueError(f"Unsupported model_type: {model_type}. Available: {list(MODEL_CONFIGS.keys())} or None")

        self.predictor = None
        if "sam3" in model_type:
            self.model = None
            self.processor = None
            self.cached_vision_embeds = None
            self.cached_original_sizes = None
        else:
            self.model_path = f"{MODEL_DIRECTORY}/{model_name}"
            self.model = UltralyticsSAM(self.model_path)
            if self.DEVICE in ["cpu", "cuda"]:
                self.model.to(self.DEVICE)
        if self.DEVICE == "cuda":
            torch.cuda.empty_cache()

    def _init_predictor(self, imgsz):
        self.model = None
        if self.DEVICE == "cuda":
            torch.cuda.empty_cache()
        overrides = {"model": self.model_path, "imgsz": imgsz, "save": False, "conf": 0, "device": self.DEVICE}

        is_sam2 = self.model_type.startswith("sam2")
        PredictorClass = SAM2Predictor if is_sam2 else Predictor

        if self.predictor is not None:
            if isinstance(imgsz, int):
                imgsz = [imgsz, imgsz]
            if list(imgsz) != self.predictor.imgsz:
                self.predictor = PredictorClass(overrides=overrides)
        else:
            self.predictor = PredictorClass(overrides=overrides)

    def _init_sam3(self, size: tuple[int]):
        assert isinstance(size, tuple)
        resolution = max(size)

        # Check if we need to reload
        if self.model is not None:
            is_pcs = "pcs" in self.model_type
            is_pvs = "pvs" in self.model_type

            if is_pcs and isinstance(self.model, Sam3Model):
                current_size = self.model.config.vision_config.backbone_config.image_size
                if current_size == resolution:
                    return
            elif is_pvs and isinstance(self.model, Sam3TrackerModel):
                return

        # Reload
        if self.DEVICE == "cuda":
            torch.cuda.empty_cache()

        print(f"SAM3 init with {size}")
        model_name = MODEL_CONFIGS.get(self.model_type)

        if "pcs" in self.model_type:
            config = Sam3Config.from_pretrained(model_name, cache_dir=MODEL_DIRECTORY)
            config.vision_config.backbone_config.image_size = resolution
            self.model = Sam3Model.from_pretrained(model_name, config=config, cache_dir=MODEL_DIRECTORY)
            self.processor = Sam3Processor.from_pretrained(model_name, cache_dir=MODEL_DIRECTORY)
        elif "pvs" in self.model_type:
            self.model = Sam3TrackerModel.from_pretrained(model_name, cache_dir=MODEL_DIRECTORY)
            self.processor = Sam3TrackerProcessor.from_pretrained(model_name, cache_dir=MODEL_DIRECTORY)

        self.model.eval()
        self.model.to(self.DEVICE)
        self.cached_vision_embeds = None
        self.cached_original_sizes = None
        self.last_image_hash = None
        self.text_embedding_cache = {}

    def segment_points(self, image: np.ndarray, positive_points: list[list[int]], negative_points: list[list[int]] = [], size: int | tuple[int, int] = 1024, category_name: str | None = None):
        """
        Segments an area in the image based on input points.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            positive_points (list[list[int]]): A list of [x, y] coordinates representing positive foreground points.
            negative_points (list[list[int]], optional): A list of [x, y] coordinates representing negative background points. Defaults to [].
            size (int | tuple[int, int], optional): The inference size for the model. Can be an int for square resizing or a tuple (height, width). Defaults to 1024.
            category_name (str | None, optional): The category name for future models. Defaults to None.

        Returns:
            torch.Tensor: A tensor containing the predicted masks. If `multi_mask_output` is True (i.e., only one positive point and no negative points), it returns 3 masks sorted by IoU score (tightest to loosest fit). Otherwise, it returns a single best mask.
                          The shape of the tensor is (num_masks, H, W), where H and W are the dimensions of the mask.
        """
        image = np.array(image)
        if self.model_type is None:
            raise ValueError("Segmenter was initialized with model_type=None; no model available for segmentation.")

        if "sam3" in self.model_type:
            self._init_sam3(size)

            hash_obj = hashlib.sha256(image.tobytes()).hexdigest()
            if hash_obj != self.last_image_hash:
                if "pcs" in self.model_type:
                    # Cache vision embeddings
                    image_inputs = self.processor(images=image, size=size, return_tensors="pt")
                    self.cached_original_sizes = image_inputs["original_sizes"]
                    with torch.no_grad():
                        self.cached_vision_embeds = self.model.get_vision_features(pixel_values=image_inputs.pixel_values.to(self.DEVICE))
                elif "pvs" in self.model_type:
                    image_inputs = self.processor(images=image, return_tensors="pt")
                    self.cached_original_sizes = image_inputs["original_sizes"]
                    with torch.no_grad():
                        self.cached_vision_embeds = self.model.get_image_embeddings(pixel_values=image_inputs.pixel_values.to(self.DEVICE))
                self.last_image_hash = hash_obj
            else:
                print("using cached image embedding")

            if "pcs" in self.model_type:
                all_points = copy.deepcopy(positive_points)
                all_labels = [1] * len(positive_points)
                if negative_points:
                    all_points.extend(negative_points)
                    all_labels.extend([0] * len(negative_points))

                # Convert points to boxes for SAM3 (Sam3Processor does not support input_points)
                input_boxes_list = []
                input_boxes_labels_list = []
                for p, l in zip(all_points, all_labels):
                    input_boxes_list.append([p[0], p[1], p[0], p[1]])
                    input_boxes_labels_list.append(l)

                input_boxes = [input_boxes_list]
                input_boxes_labels = [input_boxes_labels_list]

                if isinstance(size, int):
                    h, w = size, size
                else:
                    h, w = size

                text_embeds = None
                text_attention_mask = None

                if category_name:
                    if category_name in self.text_embedding_cache:
                        text_embeds, text_attention_mask = self.text_embedding_cache[category_name]
                    else:
                        text_inputs = self.processor(text=[category_name], return_tensors="pt").to(self.DEVICE)
                        with torch.no_grad():
                            text_embeds = self.model.get_text_features(**text_inputs)
                        text_attention_mask = text_inputs.attention_mask
                        self.text_embedding_cache[category_name] = (text_embeds, text_attention_mask)

                inputs = self.processor(
                    input_boxes=input_boxes,
                    input_boxes_labels=input_boxes_labels,
                    original_sizes=self.cached_original_sizes,
                    size={"height": h, "width": w},
                    return_tensors="pt",
                ).to(self.DEVICE)

                # Remove pixel_values if present (should not be, but just in case)
                if "pixel_values" in inputs:
                    del inputs["pixel_values"]

                # Remove text related inputs if we are using cached embeddings
                if text_embeds is not None:
                    if "input_ids" in inputs:
                        del inputs["input_ids"]
                    if "attention_mask" in inputs:
                        del inputs["attention_mask"]

                with torch.no_grad():
                    outputs = self.model(vision_embeds=self.cached_vision_embeds, text_embeds=text_embeds, attention_mask=text_attention_mask, **inputs)

                target_sizes = [image.shape[:2]]
                results = self.processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5, target_sizes=target_sizes)
                return results[0]["masks"]

            elif "pvs" in self.model_type:
                all_points = copy.deepcopy(positive_points)
                all_labels = [1] * len(positive_points)
                if negative_points:
                    all_points.extend(negative_points)
                    all_labels.extend([0] * len(negative_points))

                # Structure for processor: batch 1, object 1, points N, coords 2
                input_points = [[all_points]]
                input_labels = [[all_labels]]

                inputs = self.processor(input_points=input_points, input_labels=input_labels, original_sizes=self.cached_original_sizes, return_tensors="pt").to(self.DEVICE)

                # Remove pixel_values if present
                if "pixel_values" in inputs:
                    del inputs["pixel_values"]

                multimask_output = len(positive_points) == 1 and not negative_points

                with torch.no_grad():
                    outputs = self.model(image_embeddings=self.cached_vision_embeds, multimask_output=multimask_output, **inputs)

                masks = self.processor.post_process_masks(outputs.pred_masks, self.cached_original_sizes)[0]
                # masks shape: (1, N, H, W) -> (N, H, W)
                return masks[0]

        else:  # SAM 2 with ultralytics
            self._init_predictor(size)
            if self.predictor is None:
                raise RuntimeError("Predictor not initialized for Ultralytics model.")

            hash_obj = hashlib.sha256(image.tobytes()).hexdigest()
            if hash_obj != self.last_image_hash:
                self.predictor.set_image([image])
            self.last_image_hash = hash_obj

            all_points = copy.deepcopy(positive_points)
            all_labels = [1] * len(positive_points)

            if negative_points:
                all_points.extend(negative_points)
                all_labels.extend([0] * len(negative_points))

            multi_mask_output = len(positive_points) == 1

            results = self.predictor(points=[all_points], point_labels=[all_labels], imgsz=size, multimask_output=multi_mask_output)
            assert hasattr(results[0], "masks") and results[0].masks is not None
            masks = results[0].masks.data  # shape: (num_masks, H, W)
            assert multi_mask_output is False or masks.shape[0] == 3
            assert multi_mask_output is True or masks.shape[0] == 1

            iou_scores = results[0].boxes.conf
            sorted_indices = torch.argsort(iou_scores, descending=True)

            if len(positive_points) == 1 and not negative_points:
                assert masks.ndim == 3, f"Expected masks to have 3 dimensions, got {masks.ndim}"
                assert masks.shape[0] == 3, f"Expected 3 masks (tight->loose), got {masks.shape[0]}"
                return masks[sorted_indices]  # Returns the 3 masks where tightest fit is a index 0 and loosest at index 2
            else:
                return masks[0].unsqueeze(0)  # Return only the best mask

    def segment_boxes(self, image: np.ndarray, bboxes: torch.Tensor, size: int = 1024, category_labels: list[str] | None = None):
        """
        Segments areas in a image based on corresponding bounding boxes.

        Args:
            image: A numpy array representing the image.
            bboxes: A torch.Tensor of shape (num_boxes, 4), where each row is [x1, y1, x2, y2].
                The tensor contains the bounding boxes for the image.
            size: Inference size for the model. Can be an int. Defaults to 1024.
            category_labels: Optional list of category names for each box (used by SAM3 Class Aware).
                Length must match num_boxes if provided.

        Returns:
            torch.Tensor: A tensor of shape (num_boxes, height, width),
            where num_boxes is the number of boxes for that image, and height/width are the mask dimensions.
        """
        if self.model is None:
            raise ValueError("Segmenter was initialized with model_type=None; no model available for segmentation.")

        if "sam3" in self.model_type:
            if isinstance(size, int):
                size_tuple = (size, size)
            else:
                size_tuple = size
            self._init_sam3(size_tuple)

            image = np.array(image)
            hash_obj = hashlib.sha256(image.tobytes()).hexdigest()
            if hash_obj != self.last_image_hash:
                if "pcs" in self.model_type:
                    # Cache vision embeddings
                    resolution = max(size_tuple)
                    image_inputs = self.processor(images=image, size={"height": resolution, "width": resolution}, return_tensors="pt")
                    self.cached_original_sizes = image_inputs["original_sizes"]
                    with torch.no_grad():
                        self.cached_vision_embeds = self.model.get_vision_features(pixel_values=image_inputs.pixel_values.to(self.DEVICE))
                elif "pvs" in self.model_type:
                    image_inputs = self.processor(images=image, return_tensors="pt")
                    self.cached_original_sizes = image_inputs["original_sizes"]
                    with torch.no_grad():
                        self.cached_vision_embeds = self.model.get_image_embeddings(pixel_values=image_inputs.pixel_values.to(self.DEVICE))
                self.last_image_hash = hash_obj
            else:
                print("using cached image embedding")

            # Prepare boxes
            if isinstance(bboxes, torch.Tensor):
                boxes_list = bboxes.tolist()
            else:
                boxes_list = bboxes

            if not boxes_list:
                return torch.empty((0, image.shape[0], image.shape[1]))

            input_boxes = [boxes_list]  # Batch size 1

            if "pcs" in self.model_type:
                input_boxes_labels = [[1] * len(boxes_list)]

                if isinstance(size, int):
                    h, w = size, size
                else:
                    h, w = size

                inputs = self.processor(
                    input_boxes=input_boxes,
                    input_boxes_labels=input_boxes_labels,
                    original_sizes=self.cached_original_sizes,
                    size={"height": h, "width": w},
                    return_tensors="pt",
                ).to(self.DEVICE)

                # Remove pixel_values if present
                if "pixel_values" in inputs:
                    del inputs["pixel_values"]

                with torch.no_grad():
                    outputs = self.model(vision_embeds=self.cached_vision_embeds, **inputs)

                target_sizes = [image.shape[:2]]
                results = self.processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5, target_sizes=target_sizes)
                return results[0]["masks"]

            elif "pvs" in self.model_type:
                inputs = self.processor(input_boxes=input_boxes, original_sizes=self.cached_original_sizes, return_tensors="pt").to(self.DEVICE)

                # Remove pixel_values if present
                if "pixel_values" in inputs:
                    del inputs["pixel_values"]

                with torch.no_grad():
                    outputs = self.model(image_embeddings=self.cached_vision_embeds, multimask_output=False, **inputs)

                masks = self.processor.post_process_masks(outputs.pred_masks, self.cached_original_sizes)[0]
                # masks shape: (N, 1, H, W) -> (N, H, W)
                return masks.squeeze(1)

        else:
            # Ultralytics SAM/SAM2 fallback
            # boxes = bboxes.cpu().tolist() if isinstance(bboxes, torch.Tensor) else bboxes
            results = self.model.predict(image, bboxes=bboxes, imgsz=size)
            assert hasattr(results[0], "masks") and results[0].masks is not None
            masks = results[0].masks.data  # shape: (num_boxes, H, W)
            return masks

    def segment_batched_boxes(self, images: list[np.ndarray], bboxes_list: list[torch.Tensor], size: int = 1024, category_labels_list: list[list[str]] | None = None):
        """
        Batch segment multiple images using their respective bounding boxes.

        Args:
            images: List of images as numpy arrays.
            bboxes_list: List of tensors, each of shape (num_boxes_i, 4) for the corresponding image.
            size: Inference size (imgsz) passed through to the underlying model.
            category_labels_list: Optional list of category label lists, one per image (used by SAM3 Class Aware).
                Each inner list should have length matching the corresponding bboxes.

        Returns:
            List[torch.Tensor]: For each image, a tensor of shape (num_boxes_i, H, W) with the masks.
        """
        if self.model_type is None:
            raise ValueError("Segmenter was initialized with model_type=None; no model available for segmentation.")

        if len(images) != len(bboxes_list):
            raise ValueError(f"images and bboxes_list must have the same length, got {len(images)} and {len(bboxes_list)}")

        if "sam3" in self.model_type:
            if isinstance(size, int):
                size_tuple = (size, size)
            else:
                size_tuple = size
            self._init_sam3(size_tuple)

            input_boxes = []
            for boxes in bboxes_list:
                if isinstance(boxes, torch.Tensor):
                    input_boxes.append(boxes.tolist())
                else:
                    input_boxes.append(boxes)

            valid_indices = [i for i, boxes in enumerate(input_boxes) if len(boxes) > 0]
            batched_masks: list[torch.Tensor] = [torch.empty((0, 0, 0)) for _ in images]

            if not valid_indices:
                return batched_masks

            valid_images = [images[i] for i in valid_indices]
            valid_input_boxes = [input_boxes[i] for i in valid_indices]

            if "pcs" in self.model_type:
                valid_input_boxes_labels = [[1] * len(boxes) for boxes in valid_input_boxes]

                if isinstance(size, int):
                    h, w = size, size
                else:
                    h, w = size

                inputs = self.processor(images=valid_images, input_boxes=valid_input_boxes, input_boxes_labels=valid_input_boxes_labels, size={"height": h, "width": w}, return_tensors="pt").to(self.DEVICE)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                target_sizes = [img.shape[:2] for img in valid_images]
                results = self.processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5, target_sizes=target_sizes)

                for idx, res in zip(valid_indices, results):
                    batched_masks[idx] = res["masks"]

            elif "pvs" in self.model_type:
                inputs = self.processor(images=valid_images, input_boxes=valid_input_boxes, return_tensors="pt").to(self.DEVICE)

                with torch.no_grad():
                    outputs = self.model(**inputs, multimask_output=False)

                masks_list = self.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])

                for idx, masks in zip(valid_indices, masks_list):
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        batched_masks[idx] = masks.squeeze(1)
                    else:
                        batched_masks[idx] = masks

            return batched_masks

        else:
            # Ultralytics SAM/SAM2 fallback
            # Prepare indices for images that actually have boxes
            valid_indices = [i for i, boxes in enumerate(bboxes_list) if isinstance(boxes, torch.Tensor) and boxes.numel() > 0]

            # Pre-fill with empty tensors for images without boxes
            batched_masks: list[torch.Tensor] = [torch.empty((0, 0, 0)) for _ in images]

            if not valid_indices:
                return batched_masks

            # Gather valid images and their corresponding bboxes preserving order
            valid_images = [images[i] for i in valid_indices]
            valid_bboxes = [bboxes_list[i] for i in valid_indices]

            # Single batched predict call. Ultralytics SAM returns a list of results, one per image
            results = self.model.predict(valid_images, bboxes=valid_bboxes, imgsz=size)

            # Map results back to original indices
            for idx, res in zip(valid_indices, results):
                assert hasattr(res, "masks") and res.masks is not None, "Expected masks in results"
                data = res.masks.data
                print(type(data))
                if not isinstance(data, torch.Tensor):
                    try:
                        data = torch.as_tensor(data)
                    except Exception:
                        data = torch.empty((0, 0, 0))
                batched_masks[idx] = data  # (num_boxes_i, H, W)

            return batched_masks
