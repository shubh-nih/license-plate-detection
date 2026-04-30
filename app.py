import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import os

st.set_page_config(
    page_title = "License Plate Detection",
    page_icon = "🚗",
    layout = "wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Anton&display=swap');

html, body, [class*="css"] {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-family: 'Anton', sans-serif;
    font-size: 72px;
    color: white;
    letter-spacing: 2px;
    margin-top: -20px;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #A0A0A0;
    font-size: 18px;
    margin-bottom: 40px;
}

.stFileUploader {
    border: 2px dashed #444;
    padding: 20px;
    border-radius: 15px;
    background-color: #161B22;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="title">LICENSE PLATE DETECTION</div>
    <div class="subtitle">
        YOLOv11 Powered Real Time Detection System
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner = "Loading detection model…")
def load_model():
    model = YOLO("yolo11s-model.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "mov", "mkv"]
)

def process_image(image):
    image_np = np.array(image)
    results = model.predict(
        source = image_np,
        conf = 0.2,
        save = False
    )
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )
    return annotated_frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        results = model.predict(
            source = frame,
            conf = 0.2,
            imgsz = 320,
            verbose = False
        )

        annotated_frame = results[0].plot()

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        stframe.image(
            annotated_frame,
            channels = "RGB",
            width = 'stretch'
        )
    cap.release()

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, width = 'stretch')
        with st.spinner("Detecting License Plates..."):
            output_image = process_image(image)
        with col2:
            st.subheader("Detected Output")
            st.image(output_image, width = 'stretch')
            
    elif file_extension in ["mp4", "mov", "mkv"]:
        with tempfile.NamedTemporaryFile(delete = False) as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name
        st.subheader("Uploaded Video")
        st.video(temp_video_path)
        st.subheader("Detection Output")
        with st.spinner("Processing Video..."):
            process_video(temp_video_path)
        os.remove(temp_video_path)