from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import cv2, numpy as np, supervision as sv
from ultralytics import YOLO
import os, time, uuid

app = FastAPI(title="Surveillance API", description="Systeme de surveillance intelligente avec YOLOv8", version="1.0.0")
model = YOLO("yolov8s.pt")
resultats_store = {}
os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

@app.get("/")
def root():
    return {"message": "Surveillance API operationnelle"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8s", "timestamp": time.time()}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    input_path = f"uploads/{job_id}_{file.filename}"
    output_path = f"outputs/{job_id}_annote.mp4"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    tracker_local = sv.ByteTrack()
    line_counter = sv.LineZone(start=sv.Point(w//2, 0), end=sv.Point(w//2, h))
    frame_count, ids_vus, debut = 0, set(), time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = tracker_local.update_with_detections(detections)
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else np.array([])
        ids_vus.update(tracker_ids.tolist())
        line_counter.trigger(detections)
        labels = [f"#{tid}" for tid in tracker_ids]
        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        writer.write(annotated)
        frame_count += 1
    cap.release()
    writer.release()
    duree = round(time.time() - debut, 2)
    resultat = {"job_id": job_id, "fichier": file.filename, "frames_traitees": frame_count,
                "personnes_uniques": len(ids_vus), "entrees": line_counter.in_count,
                "sorties": line_counter.out_count, "duree_secondes": duree,
                "fps_traitement": round(frame_count/duree, 1), "video_annotee": f"/download/{job_id}"}
    resultats_store[job_id] = resultat
    return JSONResponse(content=resultat)

@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in resultats_store:
        return JSONResponse(status_code=404, content={"error": "Job non trouve"})
    return resultats_store[job_id]

@app.get("/results")
def get_all_results():
    return {"total": len(resultats_store), "resultats": resultats_store}

@app.get("/download/{job_id}")
def download_video(job_id: str):
    path = f"outputs/{job_id}_annote.mp4"
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Video non trouvee"})
    return FileResponse(path, media_type="video/mp4")
