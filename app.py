import streamlit as st
import cv2
import time
from ultralytics import YOLO
from collections import defaultdict
import os

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Object Detection & Tracing")
st.markdown("### 🚀 AI-Powered Real-Time Detection System")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
run = st.sidebar.checkbox("Start Camera")

alert_object = st.sidebar.text_input("🔔 Alert Object")
save_frames = st.sidebar.checkbox("💾 Save Frames")
show_logs = st.sidebar.checkbox("📝 Show Detection Log")

# -----------------------------
# DATA STORAGE
# -----------------------------
object_counts = defaultdict(int)
tracked_ids = set()
detection_log = []
saved_images = []

if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

# -----------------------------
# WEBRTC VIDEO PROCESSOR (FIXED CAMERA)
# -----------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.conf = conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.track(img, persist=True, conf=self.conf)
        annotated = img.copy()

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = self.model.names[cls]

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Alert system
                if alert_object and label.lower() == alert_object.lower():
                    st.toast(f"⚠️ ALERT: {label} detected!")

                # Log detection
                detection_log.append(
                    f"{time.strftime('%H:%M:%S')} - {label} ({conf_score:.2f})"
                )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -----------------------------
# CAMERA SECTION (FIXED)
# -----------------------------
if run:
    st.success("📷 Camera running (WebRTC mode - Deployment Ready)")

    webrtc_streamer(
        key="yolo-live-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.warning("⚠️ Camera OFF")

# -----------------------------
# SAVED LOGS
# -----------------------------
st.markdown("---")
st.header("📄 Detection Logs")

if show_logs:
    st.write(detection_log[-30:] if detection_log else "No detections yet.")
else:
    st.write("Enable 'Show Detection Log' to view results.")

# -----------------------------
# REFLECTION SECTION
# -----------------------------
st.markdown("---")
st.header("💭 Reflection Questions")

st.markdown("""
- What objects were easily detected?  
- What affects detection accuracy?  
  - Lighting  
  - Motion  
  - Camera quality  
""")
