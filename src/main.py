import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from ultralytics import YOLO

from color_enhance import enhance_crosswalk, enhance_traffic_colors
from models.yolo_model import create_model


def load_model_history():
    """Load model training history"""
    history_file = Path("model_history.json")
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return {"models": [], "best_model": None, "best_accuracy": 0.0}


def save_model_history(history):
    """Save model training history"""
    with open("model_history.json", "w") as f:
        json.dump(history, f, indent=4)


def update_model_history(model_path, metrics):
    """Update model history with new training results"""
    history = load_model_history()

    # Add new model entry
    model_info = {
        "path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    history["models"].append(model_info)

    # Update best model if current model is better
    current_accuracy = metrics.get("accuracy", 0.0)
    if current_accuracy > history["best_accuracy"]:
        history["best_accuracy"] = current_accuracy
        history["best_model"] = str(model_path)

        # Copy best weights to a consistent location
        best_weights = Path("best_model.pt")
        if Path(model_path).exists():
            import shutil

            shutil.copy2(model_path, best_weights)
            print(f"\nNew best model saved to {best_weights}")

    save_model_history(history)
    return history


def print_classification_report(results):
    """Print a detailed classification report from validation results"""
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    try:
        metrics = results.box  # Get box metrics

        # Overall metrics
        overall_metrics = {
            "mAP50-95": float(metrics.map),
            "mAP50": float(metrics.map50),
            "Precision": float(metrics.mp),
            "Recall": float(metrics.mr),
        }

        print("\nOverall Performance:")
        print(
            f"- mAP50-95: {overall_metrics['mAP50-95']:.3f} ({overall_metrics['mAP50-95']*100:.1f}%)"
        )
        print(
            f"- mAP50: {overall_metrics['mAP50']:.3f} ({overall_metrics['mAP50']*100:.1f}%)"
        )
        print(
            f"- Precision: {overall_metrics['Precision']:.3f} ({overall_metrics['Precision']*100:.1f}%)"
        )
        print(
            f"- Recall: {overall_metrics['Recall']:.3f} ({overall_metrics['Recall']*100:.1f}%)"
        )

        # Per-class metrics
        class_names = [
            "crosswalk",  # class 0
            "green",  # class 1
            "no",  # class 2
            "red",  # class 3
        ]
        print("\nPer-Class Performance:")

        per_class_metrics = {}
        for i, name in enumerate(class_names):
            try:
                class_metrics = {
                    "Precision": float(metrics.p[i]) if i < len(metrics.p) else 0.0,
                    "Recall": float(metrics.r[i]) if i < len(metrics.r) else 0.0,
                    "mAP50": float(metrics.ap50[i]) if i < len(metrics.ap50) else 0.0,
                    "mAP50-95": float(metrics.ap[i]) if i < len(metrics.ap) else 0.0,
                }

                print(f"\n{name.upper()}:")
                print(f"- Precision: {class_metrics['Precision']:.3f}")
                print(f"- Recall: {class_metrics['Recall']:.3f}")
                print(f"- mAP50: {class_metrics['mAP50']:.3f}")
                print(f"- mAP50-95: {class_metrics['mAP50-95']:.3f}")

                per_class_metrics[name] = class_metrics

            except Exception as e:
                print(f"Warning: Could not compute metrics for {name}: {str(e)}")
                per_class_metrics[name] = {
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "mAP50": 0.0,
                    "mAP50-95": 0.0,
                }

        return {
            "overall": overall_metrics,
            "per_class": per_class_metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"Error in print_classification_report: {str(e)}")
        return None


def verify_dataset_structure(data_dir):
    """Verify that the dataset directory structure is correct"""
    data_dir = Path(data_dir)

    # Check train directory
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")

    # Check images and labels directories
    train_images_dir = train_dir / "images"
    train_labels_dir = train_dir / "labels"

    if not train_images_dir.exists():
        raise ValueError(f"Images directory not found: {train_images_dir}")
    if not train_labels_dir.exists():
        raise ValueError(f"Labels directory not found: {train_labels_dir}")

    # Check if there are images and labels
    images = list(train_images_dir.glob("*.[jJ][pP][gG]"))
    labels = list(train_labels_dir.glob("*.txt"))

    if not images:
        raise ValueError(f"No JPG images found in {train_images_dir}")
    if not labels:
        raise ValueError(f"No label files found in {train_labels_dir}")

    print(f"Found {len(images)} images and {len(labels)} label files in train")

    return (
        train_images_dir,
        None,
    )  # Return None for val_dir as we'll use automatic split


def letterbox_resize(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while maintaining aspect ratio.

    Args:
        image: Input image
        new_shape: Target size (height, width)
        color: Padding color

    Returns:
        Resized and padded image
    """
    # Get current shape
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute new unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Calculate padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

    # Resize
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image


def prepare_enhanced_dataset(data_dir):
    """Prepare enhanced version of the dataset"""
    try:
        # Verify dataset structure
        images_dir, _ = verify_dataset_structure(data_dir)
        enhanced_dir = images_dir.parent / "images_enhanced"

        # Create enhanced directory if it doesn't exist
        enhanced_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        print("Enhancing dataset images...")
        processed = 0
        errors = 0

        for img_path in images_dir.glob("*.[jJ][pP][gG]"):
            try:
                # Read image
                img = cv2.imread(str(img_path))

                if img is None:
                    print(f"Warning: Could not read {img_path.name}")
                    errors += 1
                    continue

                # Enhance colors and contrast
                enhanced = enhance_traffic_colors(img)  # Enhance traffic signals
                enhanced = enhance_crosswalk(enhanced)  # Enhance crosswalks

                # Resize with letterboxing
                enhanced = letterbox_resize(enhanced, (640, 640))

                # Save enhanced image
                output_path = enhanced_dir / img_path.name
                cv2.imwrite(str(output_path), enhanced)
                processed += 1
                print(f"Processed: {img_path.name}")

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                errors += 1

        print(f"\nDataset enhancement complete!")
        print(f"Successfully processed: {processed} images")
        print(f"Errors encountered: {errors} images")
        print(f"Images resized to 640x640 with aspect ratio preserved")

        return enhanced_dir

    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise


def update_data_yaml(data_yaml_path, data_dir):
    """Update data.yaml with absolute paths"""
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    # Update paths to absolute paths
    data_dir = Path(data_dir).resolve()
    data["train"] = str(data_dir / "train/images")
    data["val"] = str(data_dir / "train/images")  # Use same directory for validation

    # Save updated yaml
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def main():
    try:
        # Configuration
        data_dir = Path("Traffic.v3i.yolov8").resolve()
        data_yaml = data_dir / "data.yaml"
        num_epochs = 100
        batch_size = 16
        image_size = 640
        val_split = 0.2  # 20% for validation

        # Verify dataset exists
        if not data_dir.exists():
            raise ValueError(f"Dataset directory not found: {data_dir}")
        if not data_yaml.exists():
            raise ValueError(f"Dataset configuration not found: {data_yaml}")

        # Update data.yaml with absolute paths
        print(f"\nUpdating dataset configuration...")
        update_data_yaml(data_yaml, data_dir)

        # Verify dataset structure
        print(f"\nPreparing dataset from: {data_dir}")
        train_dir, _ = verify_dataset_structure(data_dir)

        # Enhance dataset
        enhanced_dir = train_dir.parent / "images_enhanced"
        enhanced_dir.mkdir(parents=True, exist_ok=True)

        # Process training images
        print("\nEnhancing training images...")
        processed = 0
        errors = 0

        for img_path in train_dir.glob("*.[jJ][pP][gG]"):
            try:
                # Read image
                img = cv2.imread(str(img_path))

                if img is None:
                    print(f"Warning: Could not read {img_path.name}")
                    errors += 1
                    continue

                # Enhance colors and contrast
                enhanced = enhance_traffic_colors(img)
                enhanced = enhance_crosswalk(enhanced)

                # Resize with letterboxing
                enhanced = letterbox_resize(enhanced, (640, 640))

                # Save enhanced image
                output_path = enhanced_dir / img_path.name
                cv2.imwrite(str(output_path), enhanced)
                processed += 1

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                errors += 1

        print(f"Successfully processed: {processed} images")
        print(f"Errors encountered: {errors} images")

        # Initialize YOLOv8 model
        print("\nInitializing YOLOv8 model...")
        model = create_model()

        # Determine device
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {'CUDA' if device == '0' else 'CPU'}")

        # Create runs directory if it doesn't exist
        runs_dir = Path("runs/detect")
        runs_dir.mkdir(parents=True, exist_ok=True)

        print("\nStarting training...")
        results = model.train(
            data=str(data_yaml),
            epochs=num_epochs,
            imgsz=image_size,
            batch=batch_size if device == "0" else 8,  # Smaller batch size for CPU
            patience=20,  # Early stopping patience
            save=True,  # Save best model
            device=device,
            cache=True,  # Cache images for faster training
            project="runs/detect",  # Project directory
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Run name
        )

        # Run final validation and print classification report
        print("\nRunning final validation...")
        val_results = model.val(data=str(data_yaml))
        metrics = print_classification_report(val_results)

        if metrics:
            # Update model history
            best_model_path = Path(results.save_dir) / "weights/best.pt"
            history = update_model_history(str(best_model_path), metrics)

            print("\nModel History Summary:")
            print(f"Total Models Trained: {len(history['models'])}")
            print(f"Best Model: {history['best_model']}")
            print(f"Best Accuracy: {history['best_accuracy']:.3f}")

        print("\nTraining complete! To use the model:")
        print(
            """
    from ultralytics import YOLO
    import cv2
    from color_enhance import enhance_traffic_colors, enhance_crosswalk
    
    # Load the best model
    model = YOLO('best_model.pt')  # Always use the best performing model
    
    # Load and enhance image
    image = cv2.imread('path/to/image.jpg')
    enhanced_image = enhance_crosswalk(enhance_traffic_colors(image))
    
    # Predict on enhanced image
    results = model(enhanced_image)
    
    # Process results
    for r in results:
        boxes = r.boxes  # Bounding boxes
        for box in boxes:
            cls = box.cls[0]  # Class index
            conf = box.conf[0]  # Confidence score
            class_name = ['crosswalk', 'green', 'no', 'red'][int(cls)]
            print(f"Detected {class_name} with confidence {conf:.2f}")
    """
        )

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nSystem information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Torch version: {torch.__version__}")
        print(
            f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
        raise


if __name__ == "__main__":
    main()
