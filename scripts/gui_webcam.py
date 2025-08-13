#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basit YOLO webcam GUI (Tkinter)
- Model yükle (.pt veya .onnx)
- Kamera seç, başlat/durdur
- Eşik (conf) ve çözünürlük (imgsz) ayarı
- FPS ve sınıf sayısı gösterimi
"""

import os, time, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch


def find_default_model():
    # runs/*/weights/best.pt varsa onu seç
    runs = Path("runs")
    if runs.exists():
        for p in sorted(runs.glob("**/weights/best.pt"), key=os.path.getmtime, reverse=True):
            return str(p)
        for p in sorted(runs.glob("**/weights/best.onnx"), key=os.path.getmtime, reverse=True):
            return str(p)
    return ""

def list_cameras(max_index=5):
    cams = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.read()[0]:
            cams.append(i)
        if cap is not None:
            cap.release()
    return cams or [0]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fruit Detector (YOLO11s) – Webcam")
        self.geometry("980x720")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # state
        self.model = None
        self.model_path = tk.StringVar(value=find_default_model())
        self.cam_index = tk.IntVar(value=list_cameras()[0])
        self.conf = tk.DoubleVar(value=0.35)
        self.imgsz = tk.IntVar(value=640)
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        self.running = False
        self.worker = None
        self.last_fps = tk.StringVar(value="FPS: -")
        self.last_info = tk.StringVar(value="—")

        # UI
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        # Model seçimi
        ttk.Label(top, text="Model:").pack(side=tk.LEFT)
        entry = ttk.Entry(top, textvariable=self.model_path, width=60)
        entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Seç...", command=self.browse_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Yükle", command=self.load_model).pack(side=tk.LEFT, padx=4)

        # Kamera & ayarlar
        mid = ttk.Frame(self, padding=8)
        mid.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mid, text="Kamera:").pack(side=tk.LEFT)
        self.cam_combo = ttk.Combobox(mid, width=6, state="readonly",
                                      values=list_cameras(), textvariable=self.cam_index)
        self.cam_combo.pack(side=tk.LEFT, padx=4)

        ttk.Label(mid, text="Conf:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Scale(mid, from_=0.05, to=0.9, variable=self.conf, orient=tk.HORIZONTAL, length=140)\
           .pack(side=tk.LEFT, padx=4)
        ttk.Label(mid, textvariable=tk.StringVar(value="imgsz")).pack(side=tk.LEFT, padx=(10,0))
        ttk.Combobox(mid, width=6, state="readonly",
                     values=[320, 480, 512, 640, 736], textvariable=self.imgsz)\
            .pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(mid, text="GPU (CUDA)", variable=self.use_gpu,
                        state=("normal" if torch.cuda.is_available() else "disabled")).pack(side=tk.LEFT, padx=10)

        ttk.Button(mid, text="Başlat", command=self.start).pack(side=tk.LEFT, padx=6)
        ttk.Button(mid, text="Durdur", command=self.stop).pack(side=tk.LEFT, padx=2)

        ttk.Label(mid, textvariable=self.last_fps, foreground="#0a7").pack(side=tk.RIGHT)
        ttk.Label(mid, textvariable=self.last_info, foreground="#888").pack(side=tk.RIGHT, padx=8)

        # Görüntü paneli
        self.panel = ttk.Label(self)
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Durum çubuğu
        self.status = ttk.Label(self, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.set_status("Hazır")

    def set_status(self, txt):
        self.status.config(text=txt)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Model seç (.pt / .onnx)",
            filetypes=[("YOLO model", "*.pt *.onnx"), ("Tümü", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def load_model(self):
        path = self.model_path.get().strip()
        if not path:
            messagebox.showwarning("Uyarı", "Lütfen bir model dosyası seçin (.pt veya .onnx).")
            return
        try:
            self.set_status("Model yükleniyor...")
            self.update_idletasks()
            self.model = YOLO(path)
            self.set_status(f"Model yüklendi: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Model yükleme hatası", str(e))
            self.set_status("Hata")

    def start(self):
        if self.running:
            return
        if self.model is None:
            self.load_model()
            if self.model is None:
                return
        self.running = True
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.set_status("Çıkarım başladı")

    def stop(self):
        self.running = False
        self.set_status("Durduruluyor...")

    def loop(self):
        # Kamera aç
        cap = cv2.VideoCapture(int(self.cam_index.get()), cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Kamera", "Kamera açılamadı")
            self.running = False
            return

        device = 0 if (self.use_gpu.get() and torch.cuda.is_available()) else "cpu"
        t0, frames = time.time(), 0

        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue

            # YOLO çıkarım
            try:
                res = self.model(frame,
                                 imgsz=int(self.imgsz.get()),
                                 conf=float(self.conf.get()),
                                 device=device,
                                 verbose=False)[0]
                im = res.plot()  # BGR ndarray
                nc = int(getattr(res, "boxes", []).__len__() if res.boxes is not None else 0)
                self.last_info.set(f"detections: {nc}")
            except Exception as e:
                self.last_info.set("hata")
                print("inference error:", e)
                im = frame

            # FPS
            frames += 1
            if frames % 10 == 0:
                dt = time.time() - t0
                fps = frames / max(dt, 1e-6)
                self.last_fps.set(f"FPS: {fps:.1f}")

            # Tk gösterim (BGR -> RGB -> PhotoImage)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w, max_h = 960, 540
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            # referansı sakla yoksa GC temizler
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)

        cap.release()
        self.set_status("Durduruldu")

    def on_close(self):
        self.stop()
        self.after(200, self.destroy)


if __name__ == "__main__":
    App().mainloop()
