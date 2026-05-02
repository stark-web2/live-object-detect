import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from collections import defaultdict
import av
import cv2
import time
import os

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
alert_object = st.sidebar.text_input("🔔 Alert Object")
save_frames = st.sidebar.checkbox("💾 Save Frames")
show_logs = st.sidebar.checkbox("📝 Show Detection Log")

# -----------------------------
# STATE STORAGE
# -----------------------------
if "object_counts" not in st.session_state:
    st.session_state.object_counts = defaultdict(int)

if "tracked_ids" not in st.session_state:
    st.session_state.tracked_ids = set()

if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

if "saved_images" not in st.session_state:
    st.session_state.saved_images = []

# Create folder
if not os.path.exists("saved_frames"):
    os.makedirs("saved_frames")

# -----------------------------
# VIDEO PROCESSOR
# -----------------------------
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.track(img, persist=True, conf=conf)

        annotated = img.copy()

        if results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = model.names[cls]

                # Draw box
                text = f"{label} ({conf_score:.2f})"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Tracking IDs
                if box.id is not None:
                    obj_id = int(box.id[0])

                    if obj_id not in st.session_state.tracked_ids:
                        st.session_state.tracked_ids.add(obj_id)
                        st.session_state.object_counts[label] += 1

                # ALERT
                if alert_object and label.lower() == alert_object.lower():
                    st.session_state.detection_log.append(f"⚠️ ALERT: {label}")

                # LOG
                st.session_state.detection_log.append(
                    f"{time.strftime('%H:%M:%S')} - {label} ({conf_score:.2f})"
                )

        # SAVE FRAME
        if save_frames:
            filename = f"saved_frames/frame_{int(time.time()*1000)}.jpg"
            cv2.imwrite(filename, annotated)
            st.session_state.saved_images.append(filename)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -----------------------------
# START CAMERA (WEBRTC)
# -----------------------------
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

# -----------------------------
# SIDEBAR STATS
# -----------------------------
with st.sidebar:
    st.subheader("📊 Object Counts")
    for obj, count in st.session_state.object_counts.items():
        st.write(f"{obj}: {count}")

# -----------------------------
# DOWNLOAD SECTION
# -----------------------------
st.markdown("---")
st.header("💾 Download Saved Frames")

if st.session_state.saved_images:
    for img_path in st.session_state.saved_images[-10:]:
        with open(img_path, "rb") as file:
            st.download_button(
                label=f"Download {os.path.basename(img_path)}",
                data=file,
                file_name=os.path.basename(img_path),
                mime="image/jpeg"
            )
else:
    st.write("No saved frames yet.")

# -----------------------------
# LOGS
# -----------------------------
st.markdown("---")
st.header("📄 Observation & Report")

if show_logs:
    st.subheader("📝 Detection Log")
    st.write(
        st.session_state.detection_log[-20:]
        if st.session_state.detection_log else "No detections yet."
    )

# -----------------------------
# REFLECTION
# -----------------------------
st.subheader("💭 Reflection Questions")
st.markdown("""
- What objects were easily detected?  
- What affects detection accuracy?  
  - Lighting  
  - Motion  
  - Camera quality  
""")
