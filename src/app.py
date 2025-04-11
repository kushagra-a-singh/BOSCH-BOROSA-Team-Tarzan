import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from color_enhance import enhance_crosswalk, enhance_traffic_colors

# Load environment variables from .env file
load_dotenv()

# Initialize session state for FPS calculation
if "fps_times" not in st.session_state:
    st.session_state.fps_times = deque(maxlen=30)

# Class definitions with colors
CLASSES = {
    0: {"name": "crosswalk", "color": (255, 0, 0)},  # Blue for crosswalk
    1: {"name": "green", "color": (0, 255, 0)},  # Green for green signal
    2: {"name": "no", "color": (128, 128, 128)},  # Gray for no signal
    3: {"name": "red", "color": (0, 0, 255)},  # Red for red signal
}


def load_model():
    """Load the detection model"""
    try:
        # Initialize the client with environment variables
        client = InferenceHTTPClient(
            api_url=os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com"),
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )
        return client
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def calculate_fps():
    """Calculate FPS based on timestamps"""
    if len(st.session_state.fps_times) > 1:
        fps = len(st.session_state.fps_times) / (
            st.session_state.fps_times[-1] - st.session_state.fps_times[0]
        )
        return round(fps, 1)
    return 0


def plot_confidence_distribution(detections):
    """Create a histogram of detection confidences"""
    if not detections:
        return None

    confidences = [det["confidence"] for det in detections]

    fig = go.Figure(
        data=[go.Histogram(x=confidences, nbinsx=10, name="Confidence Distribution")]
    )

    fig.update_layout(
        title="Detection Confidence Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        showlegend=False,
    )

    return fig


def process_frame(
    frame, model, conf_threshold, enhance_traffic, enhance_zebra, show_fps=False
):
    """Process a single frame"""
    start_time = time.time()

    # Image enhancement
    enhanced = frame.copy()
    if enhance_traffic:
        enhanced = enhance_traffic_colors(enhanced)
    if enhance_zebra:
        enhanced = enhance_crosswalk(enhanced)

    # Create a temporary file with proper permissions
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, enhanced)

    try:
        # Run inference using the workflow
        result = model.run_workflow(
            workspace_name="borosa",
            workflow_id="detect-count-and-visualize",
            images={"image": temp_path},
            use_cache=True,
        )

        # Process detections
        result_image = enhanced.copy()
        detections = []

        if result and len(result) > 0:
            predictions = result[0].get("predictions", {})
            if "predictions" in predictions:
                for pred in predictions["predictions"]:
                    # Get box coordinates
                    x = pred["x"]
                    y = pred["y"]
                    width = pred["width"]
                    height = pred["height"]

                    x1 = int(x - width / 2)
                    y1 = int(y - height / 2)
                    x2 = int(x + width / 2)
                    y2 = int(y + height / 2)

                    # Get class and confidence
                    class_name = pred["class"]
                    conf = pred["confidence"]

                    # Draw box
                    color = CLASSES[
                        next(
                            (
                                k
                                for k, v in CLASSES.items()
                                if v["name"] == class_name.lower()
                            ),
                            0,
                        )
                    ]["color"]
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                    # Add label
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(
                        result_image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    # Store detection
                    detections.append(
                        {
                            "class": class_name,
                            "confidence": conf,
                            "box": (x1, y1, x2, y2),
                        }
                    )

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception:
            pass  # Ignore errors during cleanup

    # Calculate inference time
    inference_time = time.time() - start_time

    # Add FPS and inference time to image
    if show_fps:
        st.session_state.fps_times.append(time.time())
        fps = calculate_fps()
        cv2.putText(
            result_image,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        result_image,
        f"Inference: {inference_time*1000:.0f}ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return result_image, detections, inference_time


def main():
    st.title("Traffic Signal and Crosswalk Detection")

    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load detection model.")
        return

    # Image enhancement options
    st.sidebar.header("Enhancement Options")
    enhance_traffic = st.sidebar.checkbox("Enhance Traffic Signals", value=True)
    enhance_zebra = st.sidebar.checkbox("Enhance Crosswalks", value=True)

    # Detection settings
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.05)

    # Input source selection
    st.sidebar.header("Input Source")
    input_source = st.sidebar.radio(
        "Select Input Source", ["Image Upload", "Webcam", "Test Dataset"]
    )

    if input_source == "Test Dataset":
        st.header("Test Dataset")
        test_dir = "Traffic.v3i.yolov8/test/images"
        if not os.path.exists(test_dir):
            st.error(f"Test directory not found: {test_dir}")
            return

        test_images = [
            f for f in os.listdir(test_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        selected_image = st.selectbox("Select test image", test_images)

        if selected_image:
            image_path = os.path.join(test_dir, selected_image)
            image = cv2.imread(image_path)
            if image is not None:
                # Process and display results
                result_image, detections, inference_time = process_frame(
                    image, model, conf_threshold, enhance_traffic, enhance_zebra, True
                )

                # Display results
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

                # Show metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Inference Time", f"{inference_time*1000:.0f}ms")
                with col2:
                    st.metric("Detections", len(detections))

                # Show confidence distribution
                conf_plot = plot_confidence_distribution(detections)
                if conf_plot:
                    st.plotly_chart(conf_plot)

                # Show detection details
                if detections:
                    st.write("### Detections")
                    for det in detections:
                        st.write(
                            f"Class: {det['class']}, Confidence: {det['confidence']:.2f}"
                        )

    elif input_source == "Image Upload":
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Read and process uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Process and display results
            result_image, detections, inference_time = process_frame(
                image, model, conf_threshold, enhance_traffic, enhance_zebra, True
            )

            # Display results
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

            # Show metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Inference Time", f"{inference_time*1000:.0f}ms")
            with col2:
                st.metric("Detections", len(detections))

            # Show confidence distribution
            conf_plot = plot_confidence_distribution(detections)
            if conf_plot:
                st.plotly_chart(conf_plot)

            # Show detection details
            if detections:
                st.write("### Detections")
                for det in detections:
                    st.write(
                        f"Class: {det['class']}, Confidence: {det['confidence']:.2f}"
                    )

    else:  # Webcam
        st.header("Webcam Feed")
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])

        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break

                # Process and display results
                result_image, detections, inference_time = process_frame(
                    frame, model, conf_threshold, enhance_traffic, enhance_zebra, True
                )

                FRAME_WINDOW.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

                # Show metrics (update less frequently to avoid flooding)
                if time.time() % 1 < 0.1:  # Update roughly every second
                    col1, col2 = st.empty(), st.empty()
                    col1.metric("Inference Time", f"{inference_time*1000:.0f}ms")
                    col2.metric("Detections", len(detections))

            cap.release()


if __name__ == "__main__":
    main()
