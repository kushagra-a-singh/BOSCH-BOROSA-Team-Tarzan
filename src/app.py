import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

from color_enhance import enhance_crosswalk, enhance_traffic_colors

# Class definitions
CLASSES = {
    0: {"name": "crosswalk", "color": (255, 0, 0)},  # Blue for crosswalk
    1: {"name": "green", "color": (0, 255, 0)},  # Green for green signal
    2: {"name": "no", "color": (128, 128, 128)},  # Gray for no signal
    3: {"name": "red", "color": (0, 0, 255)},  # Red for red signal
}


def load_model_history():
    """Load model training history"""
    history_file = Path("model_history.json")
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return {"models": [], "best_model": None, "best_accuracy": 0.0}


def get_available_models():
    """Get list of available models"""
    models = []

    # Add best model if it exists
    best_model = Path("best_model.pt")
    if best_model.exists():
        models.append(str(best_model))

    # Add models from runs directory
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        for run_dir in runs_dir.glob("train_*"):
            model_path = run_dir / "weights/best.pt"
            if model_path.exists():
                models.append(str(model_path))

    return models


def main():
    st.title("Traffic Signal and Crosswalk Detection")

    # Sidebar - Model Information
    st.sidebar.header("Model Information")

    # Load model history
    history = load_model_history()
    if history["best_model"]:
        st.sidebar.success(f"Best Model Accuracy: {history['best_accuracy']:.3f}")
        st.sidebar.info(f"Total Models Trained: {len(history['models'])}")

    # Model selection
    models = get_available_models()
    if not models:
        st.error("No trained models found. Please train a model first.")
        return

    default_model = "best_model.pt" if Path("best_model.pt").exists() else models[0]
    selected_model = st.sidebar.selectbox(
        "Select Model", models, index=models.index(default_model)
    )

    # Load selected model
    try:
        model = YOLO(selected_model)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Image enhancement options
    st.sidebar.header("Enhancement Options")
    enhance_traffic = st.sidebar.checkbox("Enhance Traffic Signals", value=True)
    enhance_zebra = st.sidebar.checkbox("Enhance Crosswalks", value=True)

    # Detection settings
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Image upload
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Enhance image if selected
        enhanced = image.copy()
        if enhance_traffic:
            enhanced = enhance_traffic_colors(enhanced)
        if enhance_zebra:
            enhanced = enhance_crosswalk(enhanced)

        # Display enhanced image
        if enhance_traffic or enhance_zebra:
            st.subheader("Enhanced Image")
            st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

        # Run detection
        results = model(enhanced, conf=conf_threshold)

        # Process and display results
        result_image = enhanced.copy()
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Draw box
                color = CLASSES[cls]["color"]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

                # Add label
                label = f"{CLASSES[cls]['name']} {conf:.2f}"
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
                        "class": CLASSES[cls]["name"],
                        "confidence": conf,
                        "box": (x1, y1, x2, y2),
                    }
                )

        # Display results
        st.subheader("Detection Results")
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

        # Display detections
        if detections:
            st.subheader("Detected Objects")
            for det in detections:
                st.write(
                    f"- {det['class'].capitalize()} (Confidence: {det['confidence']:.2f})"
                )
        else:
            st.info("No objects detected in the image.")


if __name__ == "__main__":
    main()
