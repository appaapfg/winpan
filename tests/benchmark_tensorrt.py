import cv2
from ultralytics import YOLO


def benchmark_ultralytics_model(model, num_iterations=1000):
    total_preprocess_times = []
    total_inference_times = []
    total_postprocess_times = []
    for _ in range(num_iterations):
        detections = model.predict(image, imgsz=640, verbose=False)
        # Measure and record times for each stage
        total_preprocess_times.append(detections[0].speed["preprocess"])
        total_inference_times.append(detections[0].speed["inference"])
        total_postprocess_times.append(detections[0].speed["postprocess"])

    print("AVG Preprocess times:", sum(total_preprocess_times) / len(total_preprocess_times))
    print("AVG Inference times:", sum(total_inference_times) / len(total_inference_times))
    print("AVG Postprocess times:", sum(total_postprocess_times) / len(total_postprocess_times))


image = cv2.imread("debug/images/face_4096.jpg")

model = YOLO("panopticon_models/yolo11x.pt")
print("-------- Testing vanilla model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yolo11x.pt")
exported_model = model.export(format="engine", imgsz=640, half=True)
model = YOLO(exported_model)
print("-------- testing TensorRT model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yolo11x.pt")
exported_model = model.export(format="engine", imgsz=640, dynamic=True, half=True)
model = YOLO(exported_model)
print("-------- testing dynamic TensorRT model")
benchmark_ultralytics_model(model)

model = YOLO("panopticon_models/yolo11x.pt")
exported_model = model.export(format="engine", imgsz=640, nms=True, half=True)
model = YOLO(exported_model)
print("-------- Testing TensorRT model with exported nms")
benchmark_ultralytics_model(model)
