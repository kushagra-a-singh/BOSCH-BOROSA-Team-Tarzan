import json
import os
import random
import shutil
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


def split_dataset(data_dir):
    """
    Split dataset into train, validation, and test sets.
    Train: 190 images
    Validation: 31 images
    Test: 30 images
    """
    import gc
    import os
    import time

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    # Create directories if they don't exist
    for dir_path in [train_dir, valid_dir, test_dir]:
        (dir_path / "images").mkdir(parents=True, exist_ok=True)
        (dir_path / "labels").mkdir(parents=True, exist_ok=True)

    # Get all image and label files
    source_images_dir = data_dir / "train/images"
    source_labels_dir = data_dir / "train/labels"

    # Create temporary directory to store original files
    temp_dir = data_dir / "temp"
    temp_images_dir = temp_dir / "images"
    temp_labels_dir = temp_dir / "labels"

    print("\nCreating temporary backup of original files...")
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)

    # First, move all files to temporary directory
    for file in source_images_dir.glob("*.[jJ][pP][gG]"):
        shutil.move(str(file), str(temp_images_dir / file.name))
    for file in source_labels_dir.glob("*.txt"):
        shutil.move(str(file), str(temp_labels_dir / file.name))

    # Get sorted lists of files from temporary directory
    image_files = sorted(list(temp_images_dir.glob("*.[jJ][pP][gG]")))
    label_files = sorted(list(temp_labels_dir.glob("*.txt")))

    if not image_files:
        raise ValueError(f"No images found in {temp_images_dir}")

    print(f"\nFound {len(image_files)} images and {len(label_files)} labels")

    # Set random seed for reproducibility
    random.seed(42)

    # Shuffle files while keeping image-label pairs together
    paired_files = list(zip(image_files, label_files))
    random.shuffle(paired_files)

    # Split according to specified sizes
    train_pairs = paired_files[:190]
    valid_pairs = paired_files[190:221]  # Next 31 images
    test_pairs = paired_files[221:251]  # Last 30 images

    def move_files(pairs, target_dir):
        """Move files from temp directory to target directory"""
        successful = 0
        failed = 0
        for img_path, label_path in pairs:
            try:
                # Move image
                shutil.move(str(img_path), str(target_dir / "images" / img_path.name))
                # Move corresponding label
                shutil.move(
                    str(label_path), str(target_dir / "labels" / label_path.name)
                )
                successful += 1
                print(f"Moved: {img_path.name}")
            except Exception as e:
                print(f"Error moving {img_path.name}: {e}")
                failed += 1
        return successful, failed

    # Move files to respective directories
    print("\nSplitting dataset...")

    print(f"\nMoving files to training set ({len(train_pairs)} files)")
    train_success, train_failed = move_files(train_pairs, train_dir)
    print(f"Training set: {train_success} successful, {train_failed} failed")

    print(f"\nMoving files to validation set ({len(valid_pairs)} files)")
    val_success, val_failed = move_files(valid_pairs, valid_dir)
    print(f"Validation set: {val_success} successful, {val_failed} failed")

    print(f"\nMoving files to test set ({len(test_pairs)} files)")
    test_success, test_failed = move_files(test_pairs, test_dir)
    print(f"Test set: {test_success} successful, {test_failed} failed")

    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory: {e}")

    total_failed = train_failed + val_failed + test_failed
    if total_failed > 0:
        print(f"\nError: {total_failed} files could not be moved.")
        raise RuntimeError(
            f"Dataset split incomplete: {total_failed} files could not be moved"
        )

    return train_dir, valid_dir, test_dir


def update_data_yaml(data_yaml_path, data_dir):
    """Update data.yaml with absolute paths"""
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    # Update paths to absolute paths
    data_dir = Path(data_dir).resolve()
    data["train"] = str(data_dir / "train/images")
    data["val"] = str(data_dir / "valid/images")  # Use separate validation directory
    data["test"] = str(data_dir / "test/images")  # Add test directory

    # Save updated yaml
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def main():
    try:
        # Configuration
        data_dir = Path("Traffic.v3i.yolov8").resolve()
        data_yaml = data_dir / "data.yaml"
        num_epochs = 100

        # Determine device and optimize batch size
        device = "0" if torch.cuda.is_available() else "cpu"
        if device == "0":
            # Get GPU memory in GB
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Adjust batch size based on GPU memory
            if gpu_mem >= 8:  # For GPUs with 8GB or more
                batch_size = 32
            elif gpu_mem >= 6:  # For GPUs with 6-8GB
                batch_size = 24
            else:  # For GPUs with less than 6GB
                batch_size = 16
        else:
            batch_size = 8  # For CPU

        image_size = 640  # Standard YOLO input size

        print(f"\nUsing device: {'CUDA' if device == '0' else 'CPU'}")
        if device == "0":
            print(f"GPU Memory: {gpu_mem:.1f}GB")
            print(f"Batch size: {batch_size}")

        # Verify dataset exists
        if not data_dir.exists():
            raise ValueError(f"Dataset directory not found: {data_dir}")
        if not data_yaml.exists():
            raise ValueError(f"Dataset configuration not found: {data_yaml}")

        # Split dataset into train, validation, and test sets
        print("\nSplitting dataset into train, validation, and test sets...")
        train_dir, valid_dir, test_dir = split_dataset(data_dir)

        # Update data.yaml with absolute paths
        print(f"\nUpdating dataset configuration...")
        update_data_yaml(data_yaml, data_dir)

        # Initialize YOLOv8 model with improved configuration
        print("\nInitializing YOLOv8 model...")
        model = create_model()

        print("\nStarting training...")
        results = model.train(
            data=str(data_yaml),
            epochs=num_epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            cache=True,  # Cache images for faster training
            workers=min(
                8, os.cpu_count() or 1
            ),  # Number of worker threads for data loading
            project="runs/detect",
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pretrained=True,
            optimizer="AdamW",  # Use AdamW optimizer
            lr0=0.001,  # Initial learning rate
            lrf=0.0001,  # Final learning rate
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,  # Use cosine learning rate scheduler
            mixup=0.1,
            copy_paste=0.1,
            degrees=5.0,
            translate=0.1,
            scale=0.1,
            shear=2.0,
            perspective=0.0003,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            overlap_mask=True,
            mask_ratio=4,
            single_cls=False,
            rect=True,  # Enable rectangular training
            resume=False,
            amp=True,  # Enable automatic mixed precision
            fraction=1.0,
            profile=True,  # Enable performance profiling
            multi_scale=True,  # Enable multi-scale training
            sync_bn=(
                True if torch.cuda.device_count() > 1 else False
            ),  # Sync BatchNorm for multi-GPU
            deterministic=False,  # Disable deterministic training for speed
            close_mosaic=10,  # Disable mosaic augmentation in final epochs
            plots=False,  # Disable plotting during training
            save_period=-1,  # Only save best and last models
        )

        # Run validation
        print("\nRunning validation...")
        val_results = model.val(data=str(data_yaml))

        if val_results is not None:
            metrics = {
                "map50": float(val_results.box.map50),
                "map50-95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr),
            }

            print("\nValidation Metrics:")
            print(f"mAP@0.5: {metrics['map50']:.3f}")
            print(f"mAP@0.5-0.95: {metrics['map50-95']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")

            # Update model history with validation metrics
            best_model_path = Path(results.save_dir) / "weights/best.pt"
            history = update_model_history(str(best_model_path), metrics)

            # Copy best model if it's the best so far
            if metrics["map50"] > history.get("best_accuracy", 0):
                print("\nNew best model! Copying to best_model.pt")
                shutil.copy(str(best_model_path), "best_model.pt")

        print("\nTraining complete!")

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
