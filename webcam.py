
from ultralytics import YOLO

model = YOLO("models/stationery_yolo_final.pt")

model.predict(
    source=0,
    imgsz=640,
    conf=0.25,
    iou=0.5,
    show=True
)