import os
import time

import numpy as np
import psutil
import torch
from PIL import Image

from panopticon.my_inference.detect import Detector

HAVE_CUDA = torch.cuda.is_available()
print(f"is CUDA available: {HAVE_CUDA}")

# detector = Detector("yoloe_11l", "cuda")
# detector = Detector("rt_detr_101_o365", "cuda")
# detector = Detector("dab_detr", "cuda")
# detector = Detector("dab_detr_dc5", "cuda")
# detector = Detector("yolo_12n", "cuda")
detector = Detector("yolo_12n", "openvino")
# model = YOLO("yolo12m.pt")
# model.export(format="onnx")
# model = YOLO('yolo12m.onnx')
SIZE = (640, 640)
# SIZE = 1280 + 160
# SIZE = (900, 1600)
# SIZE = (1080, 1920)


image_path = "debug/images/istockphoto-856857870-612x612.jpg"
image_path = "debug/images/cars_people_1080p.jpg"
image_path = "debug/images/women-with-cars-4k-rj-3840x2160.jpg"
image_path = "debug_images/cars_people_720p.jpg"
image_path = "debug/images/face_4096.jpg"
image = Image.open(image_path).convert("RGB")
image = np.array(image)


process = psutil.Process(os.getpid())

while True:
    mem_before = process.memory_info().rss
    if HAVE_CUDA:
        torch.cuda.synchronize()
        free_vram_before, _ = torch.cuda.mem_get_info()
    start_time = time.time()

    detections = detector.detect(image, [["person"]], threshold=0.5, size=SIZE)
    # detections = model.predict(image, conf=0.5, imgsz=SIZE, device=0)

    end_time = time.time()
    if HAVE_CUDA:
        torch.cuda.synchronize()
        free_vram_after, total_vram = torch.cuda.mem_get_info()

    mem_after = process.memory_info().rss
    print(f"Detection took: {(end_time - start_time) * 1000:.4f} milliseconds")
    # print(f"Memory usage increase: {(mem_after - mem_before) / (1024 * 1024):.4f} MB")
    # print(f"Total memory used: {mem_after / (1024 * 1024):.4f} MB")
    if HAVE_CUDA:
        vram_increase = (free_vram_before - free_vram_after) / (1024 * 1024)
        vram_total_used = (total_vram - free_vram_after) / (1024 * 1024)
        print(f"VRAM usage increase: {vram_increase:.4f} MB")
        print(f"Total VRAM used: {vram_total_used:.4f} MB")
