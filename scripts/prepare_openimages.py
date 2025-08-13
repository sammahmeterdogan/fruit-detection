#!/usr/bin/env python3
import argparse, os, shutil, sys, yaml
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.openimages as oi

SPLITS = ["train", "validation", "test"]

def ensure_clean_dir(p: Path):
    if p.exists():
        print(f"[i] Temizleniyor: {p}")
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _detect_det_field(ds: fo.Dataset) -> str | None:
    """Datasetteki Detections alanını otomatik bul."""
    # 1) Şemadan ara
    schema = ds.get_field_schema()
    for name, ftype in schema.items():
        if "Detections" in str(ftype):
            return name
    # 2) Label alanlarından birini seç (varsa)
    lfs = ds.list_label_fields()
    return lfs[0] if lfs else None

def export_split(ds: fo.Dataset, export_dir: Path, split: str, classes):
    det_field = _detect_det_field(ds)
    if not det_field:
        print(f"[!] {split}: Detections alanı bulunamadı, atlanıyor.")
        return

    # En az 1 kutusu olan ve hedef sınıflardan içeren örnekler
    view = (
        ds.exists(det_field)
          .match(F(f"{det_field}.detections").length() > 0)
          .filter_labels(det_field, F("label").is_in(classes))
    )
    if len(view) == 0:
        print(f"[!] {split}: filtre sonrası örnek kalmadı, atlanıyor.")
        return

    view.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=det_field,
        split=split,  # 'train' / 'validation' / 'test'
        classes=classes,
    )

def detect_layout_and_fix(root: Path):
    # 'validation' -> 'val'
    if (root / "images" / "validation").exists():
        shutil.move(str(root / "images" / "validation"), str(root / "images" / "val"))
    if (root / "labels" / "validation").exists():
        shutil.move(str(root / "labels" / "validation"), str(root / "labels" / "val"))
    if (root / "validation").exists() and not (root / "val").exists():
        shutil.move(str(root / "validation"), str(root / "val"))

    nested = (root / "images").exists() or (root / "labels").exists()
    splitroot = (root / "train").exists() or (root / "val").exists() or (root / "test").exists()

    if nested:
        train_path = "images/train"
        val_path   = "images/val" if (root / "images" / "val").exists() else "images/validation"
        test_path  = "images/test"
    elif splitroot:
        train_path = "train/images"
        val_path   = "val/images" if (root / "val").exists() else "validation/images"
        test_path  = "test/images"
    else:
        train_path = "images/train"; val_path = "images/val"; test_path = "images/test"
    return train_path, val_path, test_path

def count_images(root: Path, rel_path: str):
    p = root / rel_path
    return len(os.listdir(p)) if p.exists() else 0

def main():
    ap = argparse.ArgumentParser(description="Open Images V7 -> YOLO (fruits)")
    ap.add_argument("--config", default="config/oi_fruits.yaml")
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--max-per-split", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    classes = args.classes if args.classes else cfg.get("classes", [])
    max_per_split = cfg.get("max_per_split", 0) if args.max_per_split is None else args.max_per_split
    out_dir = Path(args.out or cfg.get("out_dir", "datasets/fruit_oi_yolo"))

    if not classes:
        print("Hata: en az bir sınıf belirtmelisin"); sys.exit(1)

    # sınıf doğrulama (Open Images tekil isimler)
    available = set(oi.get_classes())
    invalid = [c for c in classes if c not in available]
    if invalid:
        print(f"[!] Geçersiz sınıflar atlanacak: {invalid}")
    classes = [c for c in classes if c in available]

    ensure_clean_dir(out_dir)

    max_samples = None if not max_per_split else int(max_per_split)
    print("[i] Sınıflar:", classes)
    print("[i] Çıkış   :", out_dir)

    for split in SPLITS:
        print(f"[i] İndiriliyor: {split}")
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=classes,
            only_matching=True,
            max_samples=max_samples,
        )
        export_split(ds, out_dir, split, classes)

    # Düzeni tespit et + YAML yaz
    train_rel, val_rel, test_rel = detect_layout_and_fix(out_dir)
    root_abs = out_dir.resolve().as_posix()  # ABS path

    yaml_path = out_dir / "fruit.yaml"
    names_yaml = "\n".join([f"  {i}: {c.lower()}" for i, c in enumerate(classes)])
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# YOLO dataset config (Open Images V7'dan üretildi)
path: "{root_abs}"
train: {train_rel}
val: {val_rel}
test: {test_rel}
names:
{names_yaml}
""")

    n_tr = count_images(out_dir, train_rel)
    n_va = count_images(out_dir, val_rel)
    n_te = count_images(out_dir, test_rel)
    print("\n[✓] Veri hazır!")
    print(f"  YAML : {yaml_path}")
    print(f"  Train: {n_tr}   Val: {n_va}   Test: {n_te}")

if __name__ == "__main__":
    main()
