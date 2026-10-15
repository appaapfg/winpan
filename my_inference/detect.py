import os
from typing import Any
print('==============4==============', flush=True)
import numpy as np
import torch
import yaml
from PIL import Image
print('==============5==============', flush=True)
# from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
# from rfdetr.detr import RFDETR
RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall, RFDETR = (None, None, None, None, None, None)
# from config import DETECTION_MODELS_FILE, MODEL_DIRECTORY
from transformers import (
    ConditionalDetrImageProcessor,
    DabDetrForObjectDetection,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
    DetrForObjectDetection,
    DetrImageProcessor,
    DFineForObjectDetection,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    Owlv2ForObjectDetection,
    Owlv2Processor,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)
from transformers.models.sam3 import Sam3Config, Sam3Model, Sam3Processor
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
print('==============6==============', flush=True)
from panopticon.config import DETECTION_MODELS_FILE, MODEL_DIRECTORY
print('==============7==============', flush=True)
with open(DETECTION_MODELS_FILE) as f:
    MODEL_CONFIGS = yaml.safe_load(f)
print('==============8==============', flush=True)

LOCAL_FILES_ONLY = False


class Detector:
    SUPPORTED_MODELS = list(MODEL_CONFIGS.keys())
    print('==============9==============', flush=True)

    def __init__(self, model_type: str, device: str | None = None, all_class_names: list[str] | None = None, batch_size: int = 1, resolution: tuple[int, int] | None = None):
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cached_embeddings_map = {}
        self.embed_seq_len = 0
        self.model, self.processor = self.load_model_and_processor(resolution=resolution, all_class_names=all_class_names, batch_size=batch_size)
        self.cache_input = None

    def label_list_to_str_list(self, label_indices: list[int | str] | torch.Tensor) -> list[str]:
        """
        Converts a list or tensor of label indices to a list of corresponding string class names.

        Args:
            label_indices: A list of integers or strings, or a PyTorch 1D tensor containing label indices or names.

        Returns:
            A list of string class names. Returns "<Unknown Label ID: ...>" for indices not in id2label.
        """
        str_list = []
        for index_val in label_indices:
            # If index_val is already a string, append it directly (for open-vocabulary models or text queries)
            if isinstance(index_val, str):
                str_list.append(index_val)
                continue
            # Convert to Python int. .item() is for scalar tensors.
            idx = index_val.item() if isinstance(index_val, torch.Tensor) else int(index_val)

            if RFDETR is not None and isinstance(self.model, RFDETR):
                str_list.append(self.model.class_names[idx])
            elif self.model and hasattr(self.model, "config") and hasattr(self.model.config, "id2label") and idx in self.model.config.id2label:
                str_list.append(self.model.config.id2label[idx])
            elif self.model and hasattr(self.model, "names") and idx in self.model.names:
                str_list.append(self.model.names[idx])
            else:
                str_list.append(f"<Unknown Label ID: {idx}>")
        return str_list

    def load_model_and_processor(self, resolution: int | tuple[int, int] | None = None, batch_size: int = 1, all_class_names: list[str] | None = None):
        """
        Args:
            resolution: used for rfdetr models only to set input size. Can be int or tuple (height, width)
            batch_size: used for some models only to batch size.
            all_class_names: used for exporting open vocubulary detection models
        """
        resolution = resolution if resolution else 640

        model_id = MODEL_CONFIGS.get(self.model_type)
        if not model_id:
            if self.model_type is None:
                return None, None
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        elif self.model_type.startswith("yolo"):
            model_path = os.path.join(MODEL_DIRECTORY, model_id)
            
            # Check if model file exists locally to prevent YOLO from attempting network downloads
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"For custom trained models like 'best.pt', please copy the file to {MODEL_DIRECTORY} "
                    f"or use an absolute path in detection_models.yaml"
                )
            
            if self.model_type.startswith("yoloe"):
                if self.model_type.endswith("-od"):
                    model = YOLOE(model_id.removesuffix("-seg.pt") + ".yaml").load(model_path)
                else:
                    model = YOLOE(model_path)
                if model.names[0] == "0":
                    self.is_fine_tuned = False
                    if all_class_names:
                        model.set_classes(all_class_names, model.get_text_pe(all_class_names))
                else:
                    self.is_fine_tuned = True
            else:
                model = YOLO(model_path)

            if self.device in ["cuda", "cpu"]:
                model.to(self.device)
            elif self.device.startswith("onnx"):
                exported_path = model.export(format="onnx", batch=batch_size, half=True, dynamic=False, imgsz=resolution)
                model = YOLO(exported_path)
                if self.device.endswith("cpu"):
                    self.device = "cpu"
                elif self.device.endswith("cuda"):
                    self.device = "cuda"
            elif self.device.startswith("openvino"):
                exported_path = model.export(format="openvino", batch=batch_size, half=True, dynamic=False, imgsz=resolution)
                model = YOLO(exported_path)
                if self.device.endswith("cpu"):
                    model(device="intel:cpu")
                elif self.device.endswith("gpu"):
                    model(device="intel:gpu")
                elif self.device.endswith("npu"):
                    model(device="intel:npu")
            elif self.device in ["tensorrt"]:
                exported_path = model.export(format="engine", batch=batch_size, half=True, dynamic=False, imgsz=resolution)
                model = YOLO(exported_path)
                # model()
            processor = None
        elif self.model_type.startswith("rf_detr"):
            if self.model_type == "rf_detr_nano":
                if resolution is not None:
                    model = RFDETRNano(resolution=resolution, device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
                else:
                    model = RFDETRNano(device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
            elif self.model_type == "rf_detr_small":
                if resolution is not None:
                    model = RFDETRSmall(resolution=resolution, device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
                else:
                    model = RFDETRSmall(device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
            elif self.model_type == "rf_detr_medium":
                if resolution is not None:
                    model = RFDETRMedium(resolution=resolution, device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
                else:
                    model = RFDETRMedium(device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
            elif self.model_type == "rf_detr_base":
                if resolution is not None:
                    model = RFDETRBase(resolution=resolution, device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
                else:
                    model = RFDETRBase(device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
            elif self.model_type == "rf_detr_large":
                if resolution is not None:
                    model = RFDETRLarge(resolution=resolution, device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
                else:
                    model = RFDETRLarge(device=self.device, pretrain_weights=f"{MODEL_DIRECTORY}/{MODEL_CONFIGS[self.model_type]}")
            model.optimize_for_inference(batch_size=batch_size)
            processor = None
        else:
            model_classes = {
                "owlv2-base": (Owlv2ForObjectDetection, Owlv2Processor),
                "owlv2-large": (Owlv2ForObjectDetection, Owlv2Processor),
                "detr": (DetrForObjectDetection, DetrImageProcessor),
                "deformable_detr": (DeformableDetrForObjectDetection, DeformableDetrImageProcessor),
                "rt_detr_50": (RTDetrForObjectDetection, RTDetrImageProcessor),
                "rt_detr_18_o365": (RTDetrForObjectDetection, RTDetrImageProcessor),
                "rt_detr_50_o365": (RTDetrForObjectDetection, RTDetrImageProcessor),
                "rt_detr_101_o365": (RTDetrForObjectDetection, RTDetrImageProcessor),
                "rt_detr_v2_18": (RTDetrV2ForObjectDetection, RTDetrImageProcessor),
                "rt_detr_v2_34": (RTDetrV2ForObjectDetection, RTDetrImageProcessor),
                "rt_detr_v2_50": (RTDetrV2ForObjectDetection, RTDetrImageProcessor),
                "rt_detr_v2_101": (RTDetrV2ForObjectDetection, RTDetrImageProcessor),
                "dab_detr": (DabDetrForObjectDetection, ConditionalDetrImageProcessor),
                "dab_detr_dc5": (DabDetrForObjectDetection, ConditionalDetrImageProcessor),
                "grounding_dino_tiny": (GroundingDinoForObjectDetection, GroundingDinoProcessor),
                "grounding_dino_base": (GroundingDinoForObjectDetection, GroundingDinoProcessor),
                "dfine_nano": (DFineForObjectDetection, RTDetrImageProcessor),
                "dfine_small": (DFineForObjectDetection, RTDetrImageProcessor),
                "dfine_medium": (DFineForObjectDetection, RTDetrImageProcessor),
                "dfine_large": (DFineForObjectDetection, RTDetrImageProcessor),
                "dfine_xlarge": (DFineForObjectDetection, RTDetrImageProcessor),
                "sam3": (Sam3Model, Sam3Processor),
            }
            # Prefer explicit mapping from model_type to HF model classes
            try:
                model_class, processor_class = model_classes[self.model_type]
            except KeyError:
                # If the model_type isn't one of the known HF keys, attempt to
                # infer a local ultralytics/YOLO-style model by examining the
                # configured model_id (local filename or exported artifact).
                if isinstance(model_id, str) and any(model_id.lower().endswith(ext) for ext in (".pt", ".onnx", ".yaml", ".engine")):
                    model_path = os.path.join(MODEL_DIRECTORY, model_id)
                    
                    # Check if model file exists locally to prevent YOLO from attempting network downloads
                    if not os.path.isfile(model_path):
                        raise FileNotFoundError(
                            f"Model file not found: {model_path}\n"
                            f"For custom trained models like 'best.pt', please copy the file to {MODEL_DIRECTORY} "
                            f"or use an absolute path in detection_models.yaml"
                        )
                    
                    # Load as a YOLO model by default for local artifacts
                    model = YOLO(model_path)
                    processor = None
                    # Handle device placement/export choices similarly to the yolo branch above
                    if self.device in ["cuda", "cpu"]:
                        model.to(self.device)
                    elif self.device.startswith("onnx"):
                        exported_path = model.export(format="onnx", batch=batch_size, half=True, dynamic=False, imgsz=resolution)
                        model = YOLO(exported_path)
                        if self.device.endswith("cpu"):
                            self.device = "cpu"
                        elif self.device.endswith("cuda"):
                            self.device = "cuda"
                    elif self.device.startswith("openvino"):
                        exported_path = model.export(format="openvino", batch=batch_size, half=True, dynamic=False, imgsz=resolution)
                        model = YOLO(exported_path)
                        if self.device.endswith("cpu"):
                            model(device="intel:cpu")
                        elif self.device.endswith("gpu"):
                            model(device="intel:gpu")
                        elif self.device.endswith("npu"):
                            model(device="intel:npu")
                    elif self.device in ["tensorrt"]:
                        exported_path = model.export(format="engine", batch=batch_size, half=True, dynamic=False, imgsz=resolution)
                        model = YOLO(exported_path)

                    return model, processor
                # Unknown model_type; re-raise with clearer message
                raise KeyError(f"Unsupported model_type for HF loading: {self.model_type}")

            if self.model_type == "sam3":
                config = Sam3Config.from_pretrained(model_id, cache_dir=MODEL_DIRECTORY, local_files_only=LOCAL_FILES_ONLY)
                if resolution:
                    if isinstance(resolution, tuple):
                        config.vision_config.backbone_config.image_size = max(resolution)
                    else:
                        config.vision_config.backbone_config.image_size = resolution
                model = model_class.from_pretrained(model_id, config=config, cache_dir=MODEL_DIRECTORY, local_files_only=LOCAL_FILES_ONLY)
            else:
                model = model_class.from_pretrained(model_id, cache_dir=MODEL_DIRECTORY, local_files_only=LOCAL_FILES_ONLY)

            processor = processor_class.from_pretrained(model_id, cache_dir=MODEL_DIRECTORY, local_files_only=LOCAL_FILES_ONLY)
            model.eval()  # Explicitly set the model to evaluation mode

            if self.model_type == "sam3" and self.device == "tensorrt":
                self.device = "cuda"
                model.to(self.device)

                model.get_vision_features = torch.compile(model.get_vision_features, backend="tensorrt")
                model = torch.compile(model, backend="tensorrt")
            else:
                model.to(self.device)

            if self.model_type == "sam3" and all_class_names:
                # Compute text embeddings for all classes
                inputs = processor(text=all_class_names, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    embeds = model.get_text_features(**inputs)

                self.embed_seq_len = embeds.shape[1]
                self.cached_embeddings_map = {}
                for i, name in enumerate(all_class_names):
                    self.cached_embeddings_map[name] = (embeds[i], inputs.attention_mask[i])

        return model, processor

    def detect(self, image: Image.Image | np.ndarray | list[Image.Image | np.ndarray], detection_labels: list[list[str | int]], threshold: float = 0.4, size: tuple[int, int] = (640, 640), nms_iou: float = 0.7, verbose=True) -> list[dict]:
        assert self.model is not None

        # Handle single image input by converting it to a list
        is_single_image = not isinstance(image, list)
        images = [image] if is_single_image else image

        if not images:
            return []  # Return empty list for empty input batch

        # Ensure detection_labels is a list of lists, one for each image in the batch
        # If only one list of labels is provided, duplicate it for each image
        if len(detection_labels) == 1 and len(images) > 1:
            batch_detection_labels = detection_labels * len(images)
        elif len(detection_labels) != len(images):
            raise ValueError("Number of detection_labels lists must match the number of images or be a single list for the batch.")
        else:
            batch_detection_labels = detection_labels

        if not batch_detection_labels or not batch_detection_labels[0]:
            # If no target labels are provided, return empty results for all images
            return [{"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([])} for _ in images]

        if isinstance(self.model, (Owlv2ForObjectDetection, GroundingDinoForObjectDetection)):  # transformers open vocabulary object detector
            inputs = self.processor(text=batch_detection_labels, images=images, size={"height": size[0], "width": size[1]}, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.tensor([img.shape[:2] for img in images], device=self.device)
            results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
            return results
        elif isinstance(self.model, (YOLOE, YOLOWorld)):  # ultralytics open vocabulary object detectors
            # Only support text prompts (detection_labels)

            prompt_names = batch_detection_labels[0]
            # Set text prompt (only once is needed, but safe to call)
            if self.cache_input is None or self.cache_input != prompt_names:
                if isinstance(self.model, YOLOWorld):
                    self.model.set_classes(prompt_names)
                self.cache_input = prompt_names

            # Run detection (batch or single)
            u_label2id = {name: i for i, name in self.model.names.items()}
            ids = [u_label2id.get(label) for label in detection_labels[0]]
            ids = [id for id in ids if id is not None]
            results = self.model.predict(images, imgsz=size, classes=ids, conf=threshold, retina_masks=True, iou=nms_iou, verbose=verbose, device=self.device)

            return self.ultralytics_detection2hf_detection(results)
        elif isinstance(self.model, YOLO):  # ultralytics supervised object detector
            u_label2id = {name: i for i, name in self.model.names.items()}
            ids = [u_label2id.get(label) for label in detection_labels[0]]
            ids = [id for id in ids if id is not None]
            results = self.model.predict(images, imgsz=size, device=self.device, conf=threshold, retina_masks=True, classes=ids, iou=nms_iou, verbose=verbose)

            return self.ultralytics_detection2hf_detection(results)
        elif isinstance(  # transformers supervised object detectors
            self.model,
            (DetrForObjectDetection, DeformableDetrForObjectDetection, RTDetrForObjectDetection, RTDetrV2ForObjectDetection, DabDetrForObjectDetection, DFineForObjectDetection),
        ):
            # For these models, detection_labels are used to filter results *after* inference.
            # The model infers all possible classes it was trained on.
            # We assume detection_labels is a list containing one list of target class names for the batch.
            if not batch_detection_labels or not batch_detection_labels[0]:
                # If no target labels are provided, return empty results for all images
                return [{"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([])} for _ in images]

            target_class_names = batch_detection_labels[0]
            relevant_model_label_ids = set()
            # model_label_to_query_label = {} # This mapping is not needed for filtering

            for name in target_class_names:
                # Fallback: try model's label2id if available
                if self.model and hasattr(self.model.config, "label2id") and name in self.model.config.label2id:
                    model_id = self.model.config.label2id[name]
                    relevant_model_label_ids.add(model_id)
                else:
                    print(f"Warning: Class '{name}' not found in config or {self.model_type} model's label2id. It will be ignored.")

            if not relevant_model_label_ids:
                print(f"Warning: None of the provided labels {target_class_names} are recognized by the {self.model_type} model.")
                return [{"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([])} for _ in images]

            inputs = self.processor(images=images, size={"height": size[0], "width": size[1]}, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([img.shape[:2] for img in images], device=self.device)
            # post_process_object_detection returns a list of results, one per image
            processed_results_list = self.processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)

            final_results = []
            for processed_results in processed_results_list:
                original_model_labels = processed_results["labels"]
                original_boxes = processed_results["boxes"]
                original_scores = processed_results["scores"]

                filtered_boxes_list, filtered_scores_list, filtered_model_labels_list = (
                    [],
                    [],
                    [],
                )

                for i, model_label_id_tensor in enumerate(original_model_labels):
                    model_label_id = model_label_id_tensor.item()
                    if model_label_id in relevant_model_label_ids:
                        filtered_boxes_list.append(original_boxes[i])
                        filtered_scores_list.append(original_scores[i])
                        filtered_model_labels_list.append(model_label_id_tensor)

                if not filtered_boxes_list:
                    final_results.append(
                        {
                            "scores": torch.tensor([]),
                            "labels": torch.tensor([]),
                            "boxes": torch.tensor([]),
                        }
                    )
                else:
                    final_results.append(
                        {
                            "scores": torch.stack(filtered_scores_list),
                            "labels": torch.stack(filtered_model_labels_list),
                            "boxes": torch.stack(filtered_boxes_list),
                        }
                    )
            return final_results

        elif RFDETR is not None and isinstance(self.model, RFDETR):
            # Reload model with correct resolution if needed
            if self.model._optimized_resolution != max(size) or self.model._optimized_batch_size < len(images):
                self.model, _ = self.load_model_and_processor(resolution=max(size), batch_size=len(images))

            target_class_names = batch_detection_labels[0]
            target_class_ids = [dict(zip(self.model.class_names.values(), self.model.class_names.keys()))[label] for label in target_class_names]

            raw_predictions = self.model.predict(images, threshold=threshold)
            if not isinstance(raw_predictions, list):  # rfdetr returns scalar prediction when a single input is provided
                raw_predictions = [raw_predictions]

            filtered_predictions = [prediction[np.isin(prediction.class_id, target_class_ids)] for prediction in raw_predictions]

            final_results = []
            for prediction in filtered_predictions:
                boxes = prediction.xyxy
                scores = prediction.confidence
                text_labels = self.label_list_to_str_list(prediction.class_id.tolist())

                final_results.append(
                    {
                        "boxes": torch.tensor(boxes),
                        "scores": torch.tensor(scores),
                        "labels": torch.tensor(prediction.class_id),
                        "text_labels": text_labels,
                    }
                )

            return final_results

        elif self.model_type == "sam3":
            # Pre-compute vision embeddings for the batch
            img_inputs = self.processor(images=images, return_tensors="pt", size={"height": size[0], "width": size[1]}).to(self.device)
            with torch.no_grad():
                vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)

            target_sizes = []
            for img in images:
                if isinstance(img, Image.Image):
                    target_sizes.append((img.height, img.width))
                else:
                    target_sizes.append(img.shape[:2])

            # Initialize results containers
            image_results = [{"scores": [], "labels": [], "boxes": [], "masks": [], "text_labels": []} for _ in images]

            # Determine max number of prompts
            max_prompts = max(len(prompts) for prompts in batch_detection_labels)

            for k in range(max_prompts):
                current_prompts = []
                active_indices = []
                for i, prompts in enumerate(batch_detection_labels):
                    if k < len(prompts):
                        current_prompts.append(prompts[k])
                        active_indices.append(i)
                    else:
                        current_prompts.append("padding")  # Dummy prompt

                # Check if we can use cached embeddings
                can_use_cache = bool(self.cached_embeddings_map) and all((p in self.cached_embeddings_map or p == "padding") for p in current_prompts)

                if can_use_cache:
                    batch_embeds = []
                    batch_masks = []
                    for p in current_prompts:
                        if p == "padding":
                            # Create dummy embedding and mask
                            # Use first cached embedding to get shape/dtype/device
                            first_embed = next(iter(self.cached_embeddings_map.values()))[0]
                            hidden_dim = first_embed.shape[-1]

                            e = torch.zeros((self.embed_seq_len, hidden_dim), device=self.device, dtype=first_embed.dtype)
                            m = torch.zeros((self.embed_seq_len,), device=self.device, dtype=torch.long)
                            batch_embeds.append(e)
                            batch_masks.append(m)
                        else:
                            e, m = self.cached_embeddings_map[p]
                            batch_embeds.append(e)
                            batch_masks.append(m)

                    text_embeds = torch.stack(batch_embeds)
                    attention_mask = torch.stack(batch_masks)

                    with torch.no_grad():
                        outputs = self.model(vision_embeds=vision_embeds, text_embeds=text_embeds, attention_mask=attention_mask)
                else:
                    text_inputs = self.processor(text=current_prompts, return_tensors="pt", padding=True).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(vision_embeds=vision_embeds, **text_inputs)

                results = self.processor.post_process_instance_segmentation(outputs, threshold=threshold, mask_threshold=0.5, target_sizes=target_sizes)

                for i, result in enumerate(results):
                    if i in active_indices:
                        num = len(result["scores"])
                        if num > 0:
                            image_results[i]["scores"].append(result["scores"])
                            image_results[i]["labels"].append(torch.full((num,), k, dtype=torch.long))
                            image_results[i]["boxes"].append(result["boxes"])
                            image_results[i]["masks"].append(result["masks"])
                            image_results[i]["text_labels"].extend([current_prompts[i]] * num)

            final_results = []
            for res in image_results:
                if res["scores"]:
                    final_results.append(
                        {
                            "scores": torch.cat(res["scores"]),
                            "labels": torch.cat(res["labels"]),
                            "boxes": torch.cat(res["boxes"]),
                            "masks": torch.cat(res["masks"]),
                            "text_labels": res["text_labels"],
                        }
                    )
                else:
                    final_results.append(
                        {
                            "scores": torch.tensor([]),
                            "labels": torch.tensor([]),
                            "boxes": torch.tensor([]),
                            "masks": torch.tensor([]),
                            "text_labels": [],
                        }
                    )
            return final_results

        else:
            raise ValueError(f"Unsupported model type in detect method: {type(self.model)}")

    def set_box_prompt(self, reference_image: Image.Image | np.ndarray, coco_detections: list[dict], categories: list[dict[str, Any]], size: tuple[int, int] = (640, 640)):
        if not coco_detections:
            raise ValueError("'coco_detections' must be a non-empty list.")

        boxes = []
        classes = []
        for det in coco_detections:
            boxes.append(det.get("bbox"))
            classes.append(det.get("category_id"))

        if not boxes or not classes:
            raise ValueError("'boxes' and 'classes' must be non-empty lists.")
        if len(boxes) != len(classes):
            raise ValueError("Length of 'boxes' and 'classes' lists must match (per image).")

        if isinstance(self.model, YOLOE):
            visual_prompts = {"bboxes": boxes, "cls": classes}

            # Run inference on multiple images, using the provided visual prompts as guidance
            _ = self.model.predict(source=None, refer_image=reference_image, visual_prompts=visual_prompts, predictor=YOLOEVPSegPredictor, imgsz=size)

            # update model class names
            id2label = {cat["id"]: cat["name"] for cat in categories}
            names = [value for key, value in sorted(id2label.items())]
            self.model.set_classes(names, self.model.pe)

            print(f"visual_prompts set, {self.model.names}")
        elif isinstance(self.model, Sam3Model):
            raise NotImplementedError("SAM3 box prompt not yet implemented")
        else:
            raise TypeError("Model does not support box prompt")

    def ultralytics_detection2hf_detection(self, results):
        processed_results = []
        for result in results:
            # Assuming result object has boxes, scores, labels attributes
            boxes = result.boxes.xyxy  # Bounding boxes
            scores = result.boxes.conf  # Confidence scores
            labels = result.boxes.cls  # Class IDs
            text_labels_full = self.label_list_to_str_list(labels.tolist())

            # try to get segmentation masks if available (e.g., YOLO -seg models)
            if result.masks is not None:
                masks_tensor = result.masks.data
            else:
                masks_tensor = None

            item = {
                "scores": scores,
                "labels": labels,
                "boxes": boxes,
                "text_labels": text_labels_full,
            }
            if masks_tensor is not None:
                item["masks"] = masks_tensor
            processed_results.append(item)

        return processed_results

    def get_supported_class_names(self) -> list[str] | None:
        """Return class names supported by the loaded model, or None for open-vocabulary models.

        Returns:
            list[str] or None: If a concrete class list is available, return it. If the model
            accepts arbitrary text prompts (open-vocabulary), return None to indicate "all supported".
        """
        if self.model is None:
            return None

        # RFDETR-like models may expose class_names
        try:
            if RFDETR is not None and isinstance(self.model, RFDETR):
                return list(self.model.class_names.values()) if hasattr(self.model, "class_names") else None
        except Exception:
            pass

        # Transformers models with config.id2label
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "id2label") and isinstance(self.model.config.id2label, dict):
                id2label = self.model.config.id2label
                try:
                    return [id2label[k] for k in sorted(id2label.keys())]
                except Exception:
                    return list(id2label.values())
        except Exception:
            pass

        # Ultralytics YOLO models: names attribute maps id->name
        try:
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                return [self.model.names[k] for k in sorted(self.model.names.keys())]
        except Exception:
            pass

        # For open-vocabulary models (Owlv2, GroundingDino, YOLOE world, etc.) return None
        return None


# Example usage (optional, for illustration):
if __name__ == "__main__":
    from PIL import Image

    DEFAULT_MODEL_TYPE = "rt_detr_v2_18"

    # MODEL_TYPE is defined globally at the top of the file.
    # selected_device can be determined here or passed to the Detector
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    # selected_device = "cpu"  # Force CPU for debugging

    print("--- Example Usage ---")
    print(f"Selected device: {selected_device}")
    print(f"Global MODEL_TYPE for example: {DEFAULT_MODEL_TYPE}")

    # Instantiate the Detector
    # You can override the global MODEL_TYPE by passing it directly:
    # detector = Detector(model_type="owlv2", device=selected_device)
    detector = Detector(model_type=DEFAULT_MODEL_TYPE, device=selected_device)

    try:
        # Ensure the debug_images directory and its contents are correct
        # img_path = "debug_images/istockphoto-856857870-612x612.jpg"
        img_path = "debug_images/istockphoto-1198578368-612x612.jpg"
        img = Image.open(img_path).convert("RGB")
        print(f"Image loaded from: {img_path}")

        queries = [["car", "person", "bus"]]

        print(f"Queries for {detector.model_type}: {queries}")

        # Logging raw detections
        print(f"\n--- Logging Detections for {detector.model_type} ---")
        # The `queries` variable is already suitable for `texts_for_detection_logging`
        detections_for_logging = detector.detect(img, queries)

        for i, result in enumerate(detections_for_logging):
            current_query_list_for_logging = queries[i]
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            print(f"Results for image (index {i}) with queries/targets: {current_query_list_for_logging}")
            if not scores.nelement():  # Check if scores tensor is empty
                print("  No objects detected (or matching filter).")
                continue
            for box, score, label_idx in zip(boxes, scores, labels):
                box_coords = [round(b.item(), 2) for b in box]
                # Convert model label ID to string using the detector's method
                actual_label_text = detector.label_list_to_str_list([label_idx])[0]
                print(f"  Detected {actual_label_text} with confidence {round(score.item(), 3)} at {box_coords}")

    except FileNotFoundError as e:
        print(f"Error: Image file not found. {e}. Please ensure '{img_path}' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
