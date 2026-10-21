"""Inferencia de modelos YOLO de segmentación; cargar solo checkpoints confiables."""
from __future__ import annotations
import csv
import io
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Experimento S43": ROOT / "models" / "exp-s43.pt",
    "Experimento S44": ROOT / "models" / "exp-s44.pt",
    "Experimento V1": ROOT / "models" / "exp-v1.pt",
}

@st.cache_resource(show_spinner="Cargando modelo seleccionado...")
def load_model(name: str):
    path = MODEL_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {path}")
    return YOLO(str(path))

def process_frame(model, frame: np.ndarray, conf: float, iou: float, imgsz: int, masks: bool, boxes: bool, alpha: float):
    result = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu")[0]
    output = frame.copy()
    rows = []
    masks_list = []
    names = result.names
    if result.boxes is None:
        return output, rows, masks_list
    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    segments = result.masks.xy if result.masks is not None else []
    for i, (rect, score, cls) in enumerate(zip(xyxy, scores, classes)):
        x1, y1, x2, y2 = [int(v) for v in rect]
        mask = np.zeros(frame.shape[:2], np.uint8)
        if i < len(segments):
            poly = np.asarray(segments[i], dtype=np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly], 255)
        if masks and np.any(mask):
            tint = np.zeros_like(output)
            tint[:, :] = (74, 197, 34)  # BGR
            blended = cv2.addWeighted(output, 1 - alpha, tint, alpha, 0)
            output[mask > 0] = blended[mask > 0]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (78, 250, 164), 2)
        if boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (75, 235, 131), 2)
            label = f"{names.get(int(cls), str(cls))} {score:.0%}"
            cv2.putText(output, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (75, 235, 131), 2, cv2.LINE_AA)
        rows.append({"id": i + 1, "clase": str(names.get(int(cls), str(cls))), "confianza": round(float(score), 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area_mascara_px": int(np.count_nonzero(mask))})
        masks_list.append(mask)
    return output, rows, masks_list

def png_bytes(bgr):
    ok, enc = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen")
    return enc.tobytes()

def image_from_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Archivo de imagen ilegible o incompatible")
    return img

def zip_images(items, parameters):
    memory = io.BytesIO()
    records = []
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as z:
        for name, frame, rows, masks in items:
            safe = Path(name).stem.replace("/", "_").replace("\\", "_")
            z.writestr(f"imagenes/{safe}_resultado.png", png_bytes(frame))
            combined = np.zeros(frame.shape[:2], np.uint8)
            for idx, m in enumerate(masks, 1):
                z.writestr(f"mascaras/{safe}_bache_{idx:03d}.png", png_bytes(m))
                combined = np.maximum(combined, m)
            z.writestr(f"mascaras/{safe}_todos.png", png_bytes(combined))
            for r in rows:
                records.append({"archivo": name, **r})
        z.writestr("detecciones.json", json.dumps({"parametros": parameters, "detecciones": records}, ensure_ascii=False, indent=2))
        csv_mem = io.StringIO()
        fields = ["archivo", "id", "clase", "confianza", "x1", "y1", "x2", "y2", "area_mascara_px"]
        writer = csv.DictWriter(csv_mem, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
        z.writestr("detecciones.csv", csv_mem.getvalue().encode("utf-8-sig"))
    return memory.getvalue()

def process_video(data, original_name, model, conf, iou, imgsz, masks, boxes, alpha, frame_step, progress):
    """Procesa secuencialmente; conserva duración/FPS mediante reutilización del frame anotado."""
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / Path(original_name).name
        source.write_bytes(data)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError("No se pudo abrir el video. Pruebe un MP4 H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0: fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width < 1 or height < 1:
            cap.release(); raise ValueError("Dimensiones del video no válidas")
        raw = Path(td) / "processed_raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release(); raise RuntimeError("No se pudo crear el video de salida")
        n = 0; analyzed = 0; events = []; last_frame = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                if n % frame_step == 0 or last_frame is None:
                    last_frame, rows, _ = process_frame(model, frame, conf, iou, imgsz, masks, boxes, alpha)
                    analyzed += 1
                    for r in rows:
                        events.append({"frame": n, "segundo": round(n / fps, 3), **r})
                # Los fotogramas omitidos se conservan sin anotar; no se atribuyen detecciones no calculadas.
                writer.write(last_frame if n % frame_step == 0 else frame)
                n += 1
                if n % 8 == 0:
                    progress.progress(min(n / total, 1.0) if total > 0 else 0.0, text=f"Procesando {n}/{total if total else '?'} fotogramas")
        finally:
            cap.release(); writer.release()
        if n == 0: raise ValueError("El video no contiene fotogramas decodificables")
        progress.progress(1.0, text="Video procesado")
        playable = Path(td) / "resultado_h264.mp4"
        # H.264 para navegador cuando ffmpeg esté disponible; en caso contrario, mp4v.
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(playable)], check=True, capture_output=True, timeout=180)
            video_bytes = playable.read_bytes(); encoding = "H.264"
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            video_bytes = raw.read_bytes(); encoding = "MP4V (instale FFmpeg para H.264)"
        csv_mem = io.StringIO()
        fields = ["frame", "segundo", "id", "clase", "confianza", "x1", "y1", "x2", "y2", "area_mascara_px"]
        writer_csv = csv.DictWriter(csv_mem, fieldnames=fields); writer_csv.writeheader(); writer_csv.writerows(events)
        return {"video": video_bytes, "csv": csv_mem.getvalue().encode("utf-8-sig"), "json": json.dumps(events, ensure_ascii=False, indent=2).encode(), "frames": n, "analyzed": analyzed, "detections": len(events), "seconds": n / fps, "encoding": encoding}
