import os
import time

import cv2
import numpy as np
import psutil
import torch

from panopticon.my_inference.detect import Detector

HAVE_CUDA = torch.cuda.is_available()
print(f"is CUDA available: {HAVE_CUDA}")

process = psutil.Process(os.getpid())

MODELS = ["yolo_8n", "yolo_8s", "yolo_8m", "yolo_8l", "yolo_8x", "yolo_9t", "yolo_9s", "yolo_9m", "yolo_9c", "yolo_9e", "yolo_10n", "yolo_10s", "yolo_10m", "yolo_10b", "yolo_10l", "yolo_10x", "yolo_11n", "yolo_11s", "yolo_11m", "yolo_11l", "yolo_11x", "yolo_12n", "yolo_12s", "yolo_12m", "yolo_12l", "yolo_12x", "rt_detr_18_o365", "rt_detr_50_o365", "rt_detr_101_o365", "rf_detr_nano", "rf_detr_small", "rf_detr_medium", "grounding_dino_tiny", "grounding_dino_base", "yoloe_v8s", "yoloe_11s", "yoloe_v8m", "yoloe_11m", "yoloe_v8l", "yoloe_11l", "yolow_v2_s", "yolow_v2_m", "yolow_v2_l", "yolow_v2_x"]
MODELS = ["deformable_detr"]

NUM_ITERATIONS = 100

for model_name in MODELS:
    print(f"Loading model {model_name}")
    if HAVE_CUDA:
        torch.cuda.empty_cache()
    device = "cuda" if HAVE_CUDA else "cpu"
    try:
        detector = Detector(model_name, device="cuda")
    except torch.OutOfMemoryError:
        print(f"Model {model_name} failed to load due to memory constraints")
        continue

    # for image_path in ["debug/images/women-with-cars-4k-rj-3840x2160.jpg", "debug/images/face_4096.jpg"]:
    # for image_path in ["debug/images/women-with-cars-4k-rj-3840x2160.jpg"]:
    for image_path in ["debug/images/face_4096.jpg"]:
        image = cv2.imread(image_path)

        for resolution in [(320, 320), (480, 480), (640, 640), (960, 960), (1280, 1280), (1440, 1440)]:
            total_detection_times = []
            total_memory_used = []
            total_vram_used = []

            for i in range(NUM_ITERATIONS):
                mem_before = process.memory_info().rss
                if HAVE_CUDA:
                    torch.cuda.synchronize()
                    free_vram_before, total_vram = torch.cuda.mem_get_info()
                start_time = time.time()

                try:
                    detections = detector.detect(image, [["person"]], threshold=0.5, size=resolution, verbose=False)
                    end_time = time.time()
                    if HAVE_CUDA:
                        torch.cuda.synchronize()
                        free_vram_after, _ = torch.cuda.mem_get_info()
                    mem_after = process.memory_info().rss

                    detection_time = (end_time - start_time) * 1000
                    total_detection_times.append(detection_time)
                    total_memory_used.append(mem_after / (1024 * 1024))

                    if HAVE_CUDA:
                        vram_increase = (int(free_vram_before) - int(free_vram_after)) / (1024 * 1024)
                        vram_total_used = (int(total_vram) - int(free_vram_after)) / (1024 * 1024)
                        total_vram_used.append(vram_total_used)
                except torch.OutOfMemoryError:
                    print(f"Model {model_name} resolution {resolution} failed due to memory")
                    break

            if total_detection_times:
                print(f"\n--- {model_name}-{resolution}-{resolution} Results ---")
                print(f"Average Detection Time: {np.mean(total_detection_times):.4f} ms (Max: {np.max(total_detection_times):.4f} ms)")
                # print(f"Average Unaccounted Detection Time: {np.mean(total_unaccounted_time):.4f} ms (Max: {np.max(total_unaccounted_time):.4f} ms)")
                # print(f"Average Total Memory Used: {np.mean(total_memory_used):.4f} MB (Max: {np.max(total_memory_used):.4f} MB)")
                if HAVE_CUDA:
                    # assert np.mean(total_vram_used) == np.max(total_vram_used)
                    print(f"Average Total VRAM Used: {np.mean(total_vram_used):.4f} MB (Max: {np.max(total_vram_used):.4f} MB)")
                    # print(f"Total VRAM Used: {(total_vram_used[0]):.4f} MB")
            else:
                print(f"\n--- {model_name}-{resolution}-{resolution} Skipped due to memory ---")
