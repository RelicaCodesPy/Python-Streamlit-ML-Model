import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
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
st.caption("YOLOv8 + Streamlit WebRTC Real-Time AI Vision")

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
tracking = st.sidebar.toggle("Enable Tracking", True)
alert_person = st.sidebar.toggle("Person Alert", True)
show_count = st.sidebar.toggle("Object Counting", True)
capture = st.sidebar.button("📸 Capture Frame")

# =========================
# GLOBAL STATE
# =========================
last_capture_time = 0

# =========================
# VIDEO CALLBACK
# =========================
def video_frame_callback(frame):
    global last_capture_time

    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 480))

    # YOLO TRACKING
    results = model.track(
        img,
        persist=tracking,
        tracker="bytetrack.yaml",
        conf=confidence,
        verbose=False
    )

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

    # =========================
    # SAVE FRAME
    # =========================
    if capture and time.time() - last_capture_time > 2:
        filename = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated)
        last_capture_time = time.time()
        st.sidebar.success("Frame Saved!")

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# =========================
# START WEBCAM STREAM
# =========================
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# =========================
# FOOTER
# =========================
st.markdown("### ⚡ Powered by YOLOv8 + Streamlit + OpenCV")