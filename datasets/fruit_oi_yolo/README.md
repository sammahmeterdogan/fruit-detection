# Fruit Detection (Open Images V7 → YOLO11s → Jetson TRT)

## 1) Neler yaptık (özet)

- **Veri seti**: Open Images V7'den yalnızca meyve sınıfları (Apple, Banana, Orange, Pear, Strawberry).
- **İndirme/Export**: `scripts/prepare_openimages.py`
  - FiftyOne ile yalnız **kutulu** örnekleri çekiyor.
  - YOLOv5/Ultralytics formatına export ediyor.
  - Klasör düzenini otomatik algılıyor (nested vs splitroot) ve `fruit.yaml` yazıyor.
  - `path` **ABSOLUTE** yazıldığı için Ultralytics `dataset_dir` karışıklığı yok.
- **Hatalar ve fix’ler**
  - `Grapes` yerine Open Images tekil isim: **`Grape`**.
  - `Sample has no field 'detections'` → yalnız kutulu örnekleri filtreledik.
  - NumPy 2.x uyumsuzluğu → **`numpy==1.26.4`**.
- **Eğitim**: `scripts/train_ultralytics.py` (YOLO11s)
  - 50 epoch, `imgsz=640`, AMP açık.
  - Eğitim bitince otomatik **ONNX** export.
- **Önizleme**: `scripts/preview_dataset.py` küçük örnek görselleri `_preview/`’e kaydeder.
- **GUI**: `scripts/gui_webcam.py` basit Tk GUI; model seçimi, conf/imgsz değiştirme, webkam demo.

## 2) Kurulum (PC)

```bash
python -m pip install -U pip
python -m pip install "numpy==1.26.4" ultralytics fiftyone opencv-python pyyaml pillow
