import base64
import json
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

# Initialize the client with environment variables
client = InferenceHTTPClient(
    api_url=os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com"),
    api_key=os.getenv("ROBOFLOW_API_KEY", "fGEsh6hVqfY168zf0CBs"),
)

# Image path using proper path handling
image_path = os.path.join(
    "Traffic.v3i.yolov8",
    "train",
    "images",
    "captured_20250402_174103_jpg.rf.0dd1b4a3178fa42ad542b12a0eee5fc2.jpg",
)

# Ensure the image file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Read the image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to read image: {image_path}")

try:
    # Get predictions and visualization in one call
    result = client.run_workflow(
        workspace_name="borosa",
        workflow_id="detect-count-and-visualize",
        images={"image": image_path},
        use_cache=True,
    )

    if result and len(result) > 0:
        predictions = result[0].get("predictions", {})
        if "predictions" in predictions:
            detections = predictions["predictions"]
            print("\nDetection Results:")
            print(f"Number of objects detected: {len(detections)}")
            print("\nDetailed Predictions:")

            # Sort detections by confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)

            for pred in detections:
                print(f"Class: {pred['class']}")
                print(f"Confidence: {pred['confidence']:.2%}")
                print(f"Location: x={pred['x']:.1f}, y={pred['y']:.1f}")
                print(f"Size: width={pred['width']:.1f}, height={pred['height']:.1f}")
                print("-" * 40)

            # Print image dimensions
            if "image" in predictions:
                img_info = predictions["image"]
                print(
                    f"\nImage Dimensions: {img_info['width']}x{img_info['height']} pixels"
                )

        # Save the output image if available
        if "output_image" in result[0]:
            img_data = base64.b64decode(result[0]["output_image"])
            output_path = "detection_result.jpg"
            with open(output_path, "wb") as f:
                f.write(img_data)
            print(f"\nOutput image saved as: {output_path}")

except Exception as e:
    print(f"Error during inference: {str(e)}")
    if "result" in locals():
        print("\nFull response:")
        print(json.dumps(result, indent=2))
