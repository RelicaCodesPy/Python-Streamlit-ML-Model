import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import Counter

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(page_title="Live Object Detection & Tracking", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f2027, #203a43, #000000);
    color: #00ffff;
    font-family: monospace;
}
h1 {
    text-align: center;
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff;
}
</style>
""", unsafe_allow_html=True)

st.title("📡 Live Object Detection & Tracking System")
st.caption("YOLOv8 + Streamlit (Stable Version)")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("🎛 Control Panel")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_count = st.sidebar.toggle("Object Counting", True)
alert_person = st.sidebar.toggle("Person Alert", True)

# =========================
# INPUT MODE
# =========================
mode = st.sidebar.selectbox("Input Mode", ["Image", "Video"])

# =========================
# IMAGE MODE
# =========================
if mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model.predict(img, conf=confidence)
        annotated = results[0].plot()

        boxes = results[0].boxes

        # =========================
        # OBJECT COUNTING
        # =========================
        if boxes is not None and show_count:
            names = model.names
            counts = Counter([names[int(c)] for c in boxes.cls])

            y = 30
            for obj, num in counts.items():
                cv2.putText(annotated, f"{obj}: {num}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                y += 25

        # =========================
        # PERSON ALERT
        # =========================
        if alert_person and boxes is not None:
            if any(int(c) == 0 for c in boxes.cls):
                cv2.putText(annotated, "⚠ PERSON DETECTED",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

        st.image(annotated, channels="BGR")

# =========================
# VIDEO MODE
# =========================
elif mode == "Video":
    video_file = st.file_uploader("Upload Video", type=["mp4"])

    if video_file:
        temp_path = "temp_video.mp4"

        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        st.video(temp_path)

        cap = cv2.VideoCapture(temp_path)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            results = model.predict(frame, conf=confidence)
            annotated = results[0].plot()

            boxes = results[0].boxes

            # COUNTING
            if boxes is not None and show_count:
                names = model.names
                counts = Counter([names[int(c)] for c in boxes.cls])

                y = 30
                for obj, num in counts.items():
                    cv2.putText(annotated, f"{obj}: {num}",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                    y += 25

            # ALERT
            if alert_person and boxes is not None:
                if any(int(c) == 0 for c in boxes.cls):
                    cv2.putText(annotated, "⚠ PERSON DETECTED",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

            stframe.image(annotated, channels="BGR")

        cap.release()

# =========================
# FOOTER
# =========================
st.markdown("### ⚡ Powered by YOLOv8 + Streamlit + OpenCV")
