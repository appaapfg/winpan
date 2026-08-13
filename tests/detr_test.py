from PIL import Image

from panopticon.my_inference.deotect import Detector

detector = Detector("rt_detrv2")

image_path = "debug_images/istockphoto-856857870-612x612.jpg"
image = Image.open(image_path).convert("RGB")

while True:
    detector.detect(image)
