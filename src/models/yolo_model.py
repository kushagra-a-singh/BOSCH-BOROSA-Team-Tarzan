import os

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOModel:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def train(self, data_yaml_path, epochs=100, imgsz=640, batch=16):
        """Train the YOLOv8 model"""
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name="yolov8_traffic",
        )
        return results

    def predict(self, image_path):
        """Make predictions on a single image"""
        results = self.model(image_path)
        return results[0]

    def predict_frame(self, frame):
        """Make predictions on a video frame"""
        results = self.model(frame)
        return results[0]

    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)


def create_model(model_path=None):
    """
    Create or load a YOLOv8 model for traffic light and crosswalk detection.

    Args:
        model_path (str, optional): Path to a pre-trained model. If None, creates a new model.

    Returns:
        YOLO: A YOLOv8 model instance.
    """
    try:
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"Loaded existing model from {model_path}")
        else:
            # Use YOLOv8 model pretrained on MS COCO
            model = YOLO("yolov8x.pt")  # Using YOLOv8x which was trained on MS COCO
            model.model.nc = 4  # Set number of classes
            print(
                "Created new YOLOv8x model (MS COCO pretrained) with classes: crosswalk, green, no, red"
            )

        return model

    except Exception as e:
        print(f"Error creating/loading model: {str(e)}")
        raise
