
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11s.pt")

    model.train(
        data=r"dataset\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=30,
        cos_lr=True,

        #  augmentation
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10,
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=15,

        project="runs",
        name="stationery_yolo11s_v4",
    )
