import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


st.set_page_config(page_title="Live Object Detection & Tracking", layout="wide")

st.title("🎥 Live Object Detection & Tracing")
st.write("Turn on your webcam to detect and track objects in real time.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO tracking (this gives object IDs)
        results = self.model.track(img, persist=True, verbose=False)

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
