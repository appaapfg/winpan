import cv2
from ultralytics import YOLO


def benchmark_ultralytics_model(model, num_iterations=100):
    total_preprocess_times = []
    total_inference_times = []
    total_postprocess_times = []
    for _ in range(num_iterations):
        detections = model.predict(image, imgsz=1440, verbose=False)
        # Measure and record times for each stage
        total_preprocess_times.append(detections[0].speed["preprocess"])
        total_inference_times.append(detections[0].speed["inference"])
        total_postprocess_times.append(detections[0].speed["postprocess"])

    print("AVG Preprocess times:", sum(total_preprocess_times) / len(total_preprocess_times))
    print("AVG Inference times:", sum(total_inference_times) / len(total_inference_times))
    print("AVG Postprocess times:", sum(total_postprocess_times) / len(total_postprocess_times))


image = cv2.imread("debug/images/face_4096.jpg")

model = YOLO("panopticon_models/yoloe-11l-seg.pt")
model.set_classes(["person"])
print("-------- Testing vanilla YOLOE11l model")
benchmark_ultralytics_model(model)

model = YOLO("yoloe-11l.yaml").load("panopticon_models/yoloe-11l-seg.pt")
model.eval()
model.set_classes(["person"])
model.predict()
print("-------- Testing vanilla YOLOE11l object detection model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yolo11l-seg.pt")
print("-------- Testing vanilla YOLO11l segmentation model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yolo11l.pt")
print("-------- Testing vanilla YOLO11l model")
benchmark_ultralytics_model(model)

model = YOLO("yoloe-11l.yaml").load("panopticon_models/yoloe-11l-seg.pt")
model.eval()
model.set_classes(["person"])
model.predict()
exported_model = model.export(format="engine", dynamic=True, half=True)
model = YOLO(exported_model)
print("-------- Testing TensorRT YOLOE11l object detection model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yoloe-11l-seg.pt")
model.set_classes(["person"])
exported_model = model.export(format="engine", dynamic=True, half=True)
model = YOLO(exported_model)
print("-------- Testing TensorRT YOLOE11l segmentation model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yolo11l.pt")
exported_model = model.export(format="engine", dynamic=True, half=True)
model = YOLO(exported_model)
print("-------- Testing TensorRT YOLO11l model")
benchmark_ultralytics_model(model)
