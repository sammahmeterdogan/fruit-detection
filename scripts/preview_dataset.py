#!/usr/bin/env python3
import argparse, os, shutil, sys, yaml
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.openimages as oi

SPLITS = ["train", "validation", "test"]  # OI V7 adlandırması

def ensure_clean_dir(p: Path):
    if p.exists():
        print(f"[i] Temizleniyor: {p}")
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def export_split(ds, export_dir: Path, split: str, classes):
    # Sadece 'detections' alanı olan ve en az 1 kutu içeren örnekler
    view = ds.exists("detections")
    view = view.match(F("detections.detections").length() > 0)

    view.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections",
        split=split,
        classes=classes,
    )

def main():
    ap = argparse.ArgumentParser(description="Open Images V7 -> YOLO (fruits)")
    ap.add_argument("--config", default="config/oi_fruits.yaml",
                    help="Sınıflar / max_per_split / out_dir ayarları")
    ap.add_argument("--classes", nargs="+", help="Ayar dosyasını ezmek için: Apple Banana ...")
    ap.add_argument("--max-per-split", type=int, help="Her split için maksimum örnek (0=limitsiz)")
    ap.add_argument("--out", help="Çıkış kök klasörü (YOLO düzeni)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    classes = args.classes if args.classes else cfg.get("classes", [])
    max_per_split = cfg.get("max_per_split", 0) if args.max_per_split is None else args.max_per_split
    out_dir = Path(args.out or cfg.get("out_dir", "datasets/fruit_oi_yolo"))

    if not classes:
        print("Hata: en az bir sınıf belirtmelisin (ör. Apple Banana Orange ...)")
        sys.exit(1)

    # Geçersiz sınıf isimlerini ayıkla (Open Images resmi listesine göre)
    available = set(oi.get_classes())
    invalid = [c for c in classes if c not in available]
    if invalid:
        print(f"[!] Geçersiz sınıflar atlanacak: {invalid}")
    classes = [c for c in classes if c in available]

    # Çıkış klasörünü temizle/oluştur
    ensure_clean_dir(out_dir)

    # FiftyOne: 0/None => limitsiz
    max_samples = None if not max_per_split else int(max_per_split)
    print("[i] Sınıflar:", classes)
    print("[i] Çıkış   :", out_dir)

    for split in SPLITS:
        print(f"[i] İndiriliyor: Open Images V7 -> {split}")
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=classes,
            only_matching=True,   # yalnız seçilen sınıfların göründüğü görüntüler
            max_samples=max_samples,
        )
        export_split(ds, out_dir, split, classes)

    # validation -> val isim düzeltmesi
    def mv(s, d):
        s = out_dir / s
        d = out_dir / d
        if s.exists() and not d.exists():
            shutil.move(str(s), str(d))

    mv("images/validation", "images/val")
    mv("labels/validation", "labels/val")

    # Ultralytics uyumlu dataset yaml
    yaml_path = out_dir / "fruit.yaml"
    names_yaml = "\n".join([f"  {i}: {c.lower()}" for i, c in enumerate(classes)])
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# YOLO dataset config (Open Images V7'dan üretildi)
path: {out_dir}
train: images/train
val: images/val
test: images/test
names:
{names_yaml}
""")

    # Küçük özet
    def count_imgs(split):
        p = out_dir / "images" / split
        return len(os.listdir(p)) if p.exists() else 0

    print("\n[✓] Veri hazır!")
    print(f"  YAML : {yaml_path}")
    print(f"  Train: {count_imgs('train')}  Val: {count_imgs('val')}  Test: {count_imgs('test')}")

if __name__ == "__main__":
    main()
