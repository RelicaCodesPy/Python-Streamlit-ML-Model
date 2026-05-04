import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import numpy as np
import time

# =========================
# 🎨 TECH UI THEME
# =========================
st.set_page_config(page_title="AI Object Detection", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f2027, #203a43, #000000);
    color: #00ffff;
    font-family: 'Courier New', monospace;
}

h1 {
    text-align: center;
    color: #00ffff;
    text-shadow: 0 0 15px #00ffff;
}

.block-container {
    padding-top: 1rem;
}

.stButton>button {
    background: black;
    color: #00ffff;
    border: 1px solid #00ffff;
    border-radius: 10px;
    box-shadow: 0 0 10px #00ffff;
}

.stButton>button:hover {
    background: #00ffff;
    color: black;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Live Object Detection & Tracking")
st.write("Real-time AI detection using YOLOv8 + Webcam")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SETTINGS PANEL
# =========================
st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
track_toggle = st.sidebar.toggle("Enable Tracking", True)
save_frame = st.sidebar.button("📸 Save Snapshot")
alert_person = st.sidebar.toggle("Alert: Detect Person", False)

# =========================
# GLOBAL STATE
# =========================
frame_counter = 0
last_saved = 0

# =========================
# VIDEO CALLBACK
# =========================
def video_frame_callback(frame):
    global frame_counter, last_saved

    img = frame.to_ndarray(format="bgr24")

    # YOLO Detection + Tracking
    results = model.track(
        img,
        persist=track_toggle,
        conf=confidence,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # =========================
    # OBJECT COUNTING
    # =========================
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0

    cv2.putText(
        annotated_frame,
        f"Objects: {count}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # =========================
    # ALERT SYSTEM (PERSON)
    # =========================
    if alert_person and boxes is not None:
        for c in boxes.cls:
            if int(c) == 0:  # person
                cv2.putText(
                    annotated_frame,
                    "⚠ PERSON DETECTED!",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # =========================
    # SAVE FRAME
    # =========================
    if save_frame:
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# =========================
# START STREAM
# =========================
webrtc_streamer(
    key="ai-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown('<div class="footer">⚡ Powered by YOLOv8 | Streamlit | OpenCV</div>', unsafe_allow_html=True)
