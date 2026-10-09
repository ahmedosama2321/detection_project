
from ultralytics import YOLO

if __name__ == "__main__":

   # Evaluate the best model on the test set
    best = YOLO(r"models\stationery_yolo_final.pt")
    metrics = best.val(
        data=r"dataset\data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\n--- Test Set Results ---")
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\n--- Results for Each Class ---")
    for i, name in enumerate(best.names.values()):
        print(name, "mAP50-95:", metrics.box.maps[i])