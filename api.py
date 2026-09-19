import os
import cv2
import subprocess
import tempfile
import urllib.request
import imageio_ffmpeg
import streamlit as st
import supervision as sv
from ultralytics import YOLO

st.set_page_config(page_title="Realtime Traffic Analytics", layout="wide")
st.title("🚘 Realtime Traffic Analytics Pipeline")
st.markdown("YOLOv8 & ByteTrack tabanlı araç tespiti, takibi ve çizgi geçiş analitiği.")


# Modeli ve Varsayılan Örnek Videoyu Yükle
@st.cache_resource
def load_resources():
    model = YOLO("yolov8n.pt")
    default_video_path = "sample_traffic.mp4"
    if not os.path.exists(default_video_path):
        sample_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/raw_frames.mp4"
        try:
            urllib.request.urlretrieve(sample_url, default_video_path)
        except Exception:
            pass
    return model, default_video_path


model, default_video_path = load_resources()

# Dosya Yükleme Paneli
uploaded_file = st.file_uploader("Kendi video dosyanızı yükleyin (İsteğe Bağlı)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_input = "input_temp.mp4"
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.read())
    active_video_path = temp_input
    st.info("Yüklenen video kullanılıyor.")
elif os.path.exists(default_video_path):
    active_video_path = default_video_path
    st.info("Varsayılan örnek trafik videosu kullanılıyor.")
else:
    active_video_path = None

if active_video_path:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Girdi Videosu")
        with open(active_video_path, "rb") as f:
            st.video(f.read())

    if st.button("Analizi Başlat 🚀", type="primary"):
        with st.spinner("Video işleniyor, nesneler takip ediliyor..."):
            cap = cv2.VideoCapture(active_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

            # Çizgi Analizi Tanımı
            line_start = sv.Point(0, int(height * 0.5))
            line_end = sv.Point(width, int(height * 0.5))
            line_counter = sv.LineZone(start=line_start, end=line_end)

            tracker = sv.ByteTrack()
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            line_annotator = sv.LineZoneAnnotator()

            raw_output = "raw_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(raw_output, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                line_counter.trigger(detections=detections)

                labels = [
                    f"#{tracker_id} {model.names[class_id]}"
                    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
                ]

                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)

                out.write(annotated_frame)

            # 1. Video akışlarını güvenle kapat
            cap.release()
            out.release()

            # 2. Web uyumlu H.264 (YUV420p) formatına dönüştür
            web_output = "web_output.mp4"
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_cmd = [
                ffmpeg_exe, "-y", "-i", raw_output,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                web_output
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # 3. Çıktı videosunu ekrana bas
            with col2:
                st.subheader("İşlenmiş Çıktı Videosu")
                with open(web_output, "rb") as video_file:
                    st.video(video_file.read())

                st.success(f"Analiz Tamamlandı! 🟢 Giriş: {line_counter.in_count} | 🔴 Çıkış: {line_counter.out_count}")