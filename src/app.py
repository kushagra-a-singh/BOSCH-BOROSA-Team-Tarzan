import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

from color_enhance import enhance_crosswalk, enhance_traffic_colors

# Class definitions with colors
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


def preprocess_image(image, target_size=(640, 640)):
    """Preprocess image for model input"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate scaling factors
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)

    # Calculate new size maintaining aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas with padding
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    # Calculate padding
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2

    # Place image on canvas
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


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
        st.sidebar.success(f"Model loaded successfully: {selected_model}")

        # Show model configuration
        if st.sidebar.checkbox("Show Model Configuration"):
            st.sidebar.json(model.model.args)

        # Add model info
        st.sidebar.info(f"Model input size: {model.model.args['imgsz']}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Image enhancement options
    st.sidebar.header("Enhancement Options")
    enhance_traffic = st.sidebar.checkbox("Enhance Traffic Signals", value=True)
    enhance_zebra = st.sidebar.checkbox("Enhance Crosswalks", value=True)

    # Detection settings
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.05)

    # Debug options
    st.sidebar.header("Debug Options")
    show_debug = st.sidebar.checkbox("Show Debug Information", value=True)
    show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=True)

    # Image upload
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Error: Could not read the uploaded image")
            return

        # Display original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if show_debug:
            st.subheader("Debug Information")
            st.write(f"Original image shape: {image.shape}")
            st.write(f"Model configuration:")
            st.write(model.model.args)

        # Enhance image if selected
        enhanced = image.copy()
        if enhance_traffic:
            enhanced = enhance_traffic_colors(enhanced)
            if show_preprocessing:
                st.subheader("After Traffic Signal Enhancement")
                st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

        if enhance_zebra:
            enhanced = enhance_crosswalk(enhanced)
            if show_preprocessing:
                st.subheader("After Crosswalk Enhancement")
                st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

        # Preprocess image for model
        processed_img = preprocess_image(enhanced)

        if show_preprocessing:
            st.subheader("Preprocessed Image (Model Input)")
            st.image(processed_img)

        # Run detection with additional debugging
        try:
            if show_debug:
                st.write("Model input shape:", processed_img.shape)
                st.write("Model input type:", processed_img.dtype)
                st.write(
                    "Model input range:",
                    np.min(processed_img),
                    "-",
                    np.max(processed_img),
                )

            # Run inference
            results = model(processed_img, conf=conf_threshold, verbose=show_debug)

            if show_debug:
                st.write("Raw model output:")
                for i, r in enumerate(results):
                    st.write(f"Result {i}:")
                    st.write("- Boxes:", r.boxes)
                    st.write(
                        "- Shape:",
                        r.boxes.shape if hasattr(r.boxes, "shape") else "No shape",
                    )
                    if len(r.boxes) > 0:
                        st.write("- Classes:", r.boxes.cls)
                        st.write("- Confidences:", r.boxes.conf)

            # Process and display results
            result_image = enhanced.copy()
            detections = []

            for r in results:
                boxes = r.boxes

                if show_debug:
                    st.write(f"Processing {len(boxes)} detections")

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if show_debug:
                        st.write(
                            f"Detection: Class {cls} ({CLASSES[cls]['name']}) at {conf:.2f}"
                        )
                        st.write(f"Box: ({x1}, {y1}), ({x2}, {y2})")

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
                st.warning("No objects detected in the image.")
                st.info(
                    f"Troubleshooting steps:\n"
                    f"1. Current confidence threshold: {conf_threshold}\n"
                    f"2. Try disabling image enhancements\n"
                    f"3. Check if the model was trained properly\n"
                    f"4. Verify the model weights file\n"
                    f"5. Try retraining the model with similar images"
                )

        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            if show_debug:
                st.exception(e)


if __name__ == "__main__":
    main()
