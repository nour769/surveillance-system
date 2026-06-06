
import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tempfile, os, time
import plotly.graph_objects as go

st.set_page_config(page_title="Surveillance System", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252a3a);
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2.5rem; font-weight: 700; color: #00d4ff; }
.metric-label { font-size: 0.9rem; color: #8892b0; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    modele = st.selectbox("🤖 Modèle YOLO", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=1)
    confidence = st.slider("🎯 Seuil de confiance", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.markdown("### 📊 Benchmark")
    st.dataframe({"Modèle": ["yolov8n", "yolov8s", "yolov8m"], "FPS": [52.4, 52.8, 42.0], "Précision": [8.6, 9.9, 9.3]}, hide_index=True)

st.markdown("# 🎯 Surveillance System")
st.markdown("**Détection temps réel · Tracking · Comptage · RGPD**")
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class=\'metric-card\'><div class=\'metric-value\'>—</div><div class=\'metric-label\'>👥 Personnes uniques</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class=\'metric-card\'><div class=\'metric-value\'>—</div><div class=\'metric-label\'>🚪 Entrées</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class=\'metric-card\'><div class=\'metric-value\'>—</div><div class=\'metric-label\'>🚶 Sorties</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class=\'metric-card\'><div class=\'metric-value\'>—</div><div class=\'metric-label\'>⚡ FPS</div></div>", unsafe_allow_html=True)

st.divider()
st.markdown("### 📁 Charger une vidéo")
video_file = st.file_uploader("Glisse ta vidéo ici", type=["mp4", "avi", "mov"])

if video_file:
    st.video(video_file)
    if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        output_path = tmp_path.replace(".mp4", "_annote.mp4")

        progress = st.progress(0, text="⏳ Chargement du modèle...")
        model = YOLO(modele)
        tracker = sv.ByteTrack()
        box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5)
        trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))
        line_counter = sv.LineZone(start=sv.Point(w//2, 0), end=sv.Point(w//2, h))

        frame_count, ids_vus, detections_par_frame, debut = 0, set(), [], time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = tracker.update_with_detections(detections)
            tracker_ids = detections.tracker_id if detections.tracker_id is not None else np.array([])
            ids_vus.update(tracker_ids.tolist())
            line_counter.trigger(detections)
            detections_par_frame.append(len(detections))
            labels = [f"#{tid}" for tid in tracker_ids]
            annotated = trace_annotator.annotate(frame.copy(), detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            writer.write(annotated)
            frame_count += 1
            progress.progress(int(frame_count/max(total_frames,1)*100), text=f"🔍 Analyse... {int(frame_count/max(total_frames,1)*100)}%")

        cap.release()
        writer.release()
        duree = time.time() - debut

        progress.progress(100, text="✅ Terminé !")
        st.divider()
        st.markdown("### 📊 Résultats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Personnes uniques", len(ids_vus))
        c2.metric("🚪 Entrées", line_counter.in_count)
        c3.metric("🚶 Sorties", line_counter.out_count)
        c4.metric("⚡ FPS", round(frame_count/duree, 1))

        st.markdown("### 📈 Détections par frame")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=detections_par_frame, mode="lines", fill="tozeroy",
            line=dict(color="#00d4ff", width=2), fillcolor="rgba(0,212,255,0.1)"))
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"), height=300,
            xaxis=dict(title="Frame", gridcolor="#2d3250"),
            yaxis=dict(title="Personnes", gridcolor="#2d3250"))
        st.plotly_chart(fig, use_container_width=True)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Télécharger vidéo annotée", f.read(),
                "surveillance_annote.mp4", "video/mp4", use_container_width=True)
        os.unlink(tmp_path)
else:
    st.info("👆 Upload une vidéo pour commencer l'analyse")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔍 Détection\nYOLOv8 détecte chaque personne en temps réel")
    with c2:
        st.markdown("#### 🎯 Tracking\nID unique par personne qui persiste")
    with c3:
        st.markdown("#### 📊 Statistiques\nComptage entrées/sorties + graphique live")
