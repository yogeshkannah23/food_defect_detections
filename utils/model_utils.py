# utils/model_utils.py
from ultralytics import YOLO
import random

model = YOLO("model/best.onnx")
class_names = ['Apple', 'Rotten']
colors = {
    class_id: tuple(random.choices(range(256), k=3))
    for class_id in range(len(class_names))
}
