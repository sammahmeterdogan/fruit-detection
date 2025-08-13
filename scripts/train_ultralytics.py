from ultralytics import YOLO
from pathlib import Path

# YAML dosyasının mutlak yolu (dataset_dir karmaşasını önler)
ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "datasets" / "fruit_oi_yolo" / "fruit.yaml").as_posix()

MODEL = "yolo11s.pt"   # YOLO11s

if __name__ == "__main__":
    print(f"[i] Using data: {DATA}")
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=-1,       # otomatik batch
        device=0,       # ilk GPU
        project="runs",
        name="fruit_oi_yolo11s",
        patience=20,
        cos_lr=True
    )
    model.val(data=DATA)
    model.export(format="onnx", dynamic=True)  # Jetson için ONNX
