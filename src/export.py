import argparse

from ultralytics import YOLO


def export_model(weights_path, format="onnx"):
    """
    Export YOLOv8 model to different formats
    Supported formats:
    - onnx (default): Best for ESP32-S3 and general deployment
    - openvino: Good for Intel devices
    - engine: For NVIDIA Jetson
    - coreml: For iOS devices
    - tflite: For Android and edge devices
    """
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    print(f"Exporting model to {format} format...")

    # Export the model
    model.export(
        format=format,
        half=True,  # FP16 quantization
        int8=False,  # INT8 quantization (set to True if device supports it)
        device=0,  # GPU device
        optimize=True,  # Optimize for inference
    )

    print(
        f"Model exported successfully! Check the 'runs/detect/train/weights/' directory for the exported model."
    )

    if format == "onnx":
        print("\nFor ESP32 deployment:")
        print("1. Use the .onnx file with OpenVINO or TensorRT for inference")
        print("2. Recommended inference settings:")
        print("   - Input size: 640x640")
        print("   - FP16 precision")
        print("   - BGR input format")
        print("   - Scale factor: 1/255")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model for deployment")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to the trained weights file",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "openvino", "engine", "coreml", "tflite"],
        help="Export format",
    )

    args = parser.parse_args()
    export_model(args.weights, args.format)


if __name__ == "__main__":
    main()
