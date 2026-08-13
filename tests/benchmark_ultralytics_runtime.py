import os
import time

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

HAVE_CUDA = torch.cuda.is_available()
print(f"is CUDA available: {HAVE_CUDA}")

process = psutil.Process(os.getpid())

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e", "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x", "yoloe-v8s-seg", "yoloe-11s-seg", "yoloe-v8m-seg", "yoloe-11m-seg", "yoloe-v8l-seg", "yoloe-11l-seg", "yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2"]
MODELS = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"]
MODELS = ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
MODELS = ["yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"]

NUM_ITERATIONS = 100

for model_name in MODELS:
    print(f"Loading model {model_name}")
    if HAVE_CUDA:
        torch.cuda.empty_cache()
    model = YOLO(f"./panopticon_models/{model_name}.pt").to("cuda")
    if model_name.startswith("yoloe"):
        model.set_classes(["person"])
    model = YOLO(model.export(format="engine", half=True, dynamic=True))

    # for image_path in ["debug/images/women-with-cars-4k-rj-3840x2160.jpg", "debug/images/face_4096.jpg"]:
    for image_path in ["debug/images/face_4096.jpg"]:
        image = cv2.imread(image_path)

        for resolution in [(320, 320), (480, 480), (640, 640), (960, 960), (1280, 1280)]:
            total_detection_times = []
            total_preprocess_times = []
            total_inference_times = []
            total_postprocess_times = []
            total_memory_used = []
            total_vram_used = []
            total_unaccounted_time = []

            for i in range(NUM_ITERATIONS):
                mem_before = process.memory_info().rss
                if HAVE_CUDA:
                    torch.cuda.synchronize()
                    free_vram_before, total_vram = torch.cuda.mem_get_info()
                start_time = time.time()

                detections = model.predict(image, classes=[0], conf=0.5, imgsz=resolution, verbose=False, device=0)
                preprocess_time = detections[0].speed["preprocess"]
                inference_time = detections[0].speed["inference"]
                postprocess_time = detections[0].speed["postprocess"]

                end_time = time.time()
                if HAVE_CUDA:
                    torch.cuda.synchronize()
                    free_vram_after, _ = torch.cuda.mem_get_info()

                mem_after = process.memory_info().rss

                detection_time = (end_time - start_time) * 1000
                total_detection_times.append(detection_time)
                total_preprocess_times.append(preprocess_time)
                total_inference_times.append(inference_time)
                total_postprocess_times.append(postprocess_time)
                total_memory_used.append(mem_after / (1024 * 1024))
                total_unaccounted_time.append(detection_time - (preprocess_time + inference_time + postprocess_time))

                if HAVE_CUDA:
                    vram_increase = (int(free_vram_before) - int(free_vram_after)) / (1024 * 1024)
                    vram_total_used = (int(total_vram) - int(free_vram_after)) / (1024 * 1024)
                    total_vram_used.append(vram_total_used)

            print(f"\n--- {model_name}-{resolution} Results over {NUM_ITERATIONS} iterations  ---")
            print(f"Average Detection Time: {np.mean(total_detection_times):.4f} ms (Max: {np.max(total_detection_times):.4f} ms)")
            print(f"Average Preprocess Time: {np.mean(total_preprocess_times):.4f} ms (Max: {np.max(total_preprocess_times):.4f} ms)")
            print(f"Average Inference Time: {np.mean(total_inference_times):.4f} ms (Max: {np.max(total_inference_times):.4f} ms)")
            # print(f"Average Unaccounted Detection Time: {np.mean(total_unaccounted_time):.4f} ms (Max: {np.max(total_unaccounted_time):.4f} ms)")
            print(f"Average Postprocess Time: {np.mean(total_postprocess_times):.4f} ms (Max: {np.max(total_postprocess_times):.4f} ms)")
            # print(f"Average Total Memory Used: {np.mean(total_memory_used):.4f} MB (Max: {np.max(total_memory_used):.4f} MB)")
            if HAVE_CUDA:
                # assert np.mean(total_vram_used) == np.max(total_vram_used)
                print(f"Average Total VRAM Used: {np.mean(total_vram_used):.4f} MB (Max: {np.max(total_vram_used):.4f} MB)")
                # print(f"Total VRAM Used: {(total_vram_used[0]):.4f} MB")
