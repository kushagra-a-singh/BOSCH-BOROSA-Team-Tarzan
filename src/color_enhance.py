from pathlib import Path

import cv2
import numpy as np


def enhance_traffic_colors(image):
    """
    Enhance traffic signal colors (red and green) in the image.

    Args:
        image: Input image in BGR format

    Returns:
        Enhanced image
    """
    try:
        if image is None:
            raise ValueError("Input image is None")

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range (wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Green color range
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        # Create masks
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Enhance red and green regions
        enhanced = image.copy()

        # Enhance red
        enhanced[red_mask > 0] = [0, 0, 255]  # Pure red

        # Enhance green
        enhanced[green_mask > 0] = [0, 255, 0]  # Pure green

        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    except Exception as e:
        print(f"Error in enhance_traffic_colors: {str(e)}")
        return image


def enhance_crosswalk(image):
    """
    Enhance crosswalk patterns in the image.

    Args:
        image: Input image in BGR format

    Returns:
        Enhanced image
    """
    try:
        if image is None:
            raise ValueError("Input image is None")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply adaptive thresholding to detect zebra patterns
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Edge detection
        edges = cv2.Canny(thresh, 50, 150)

        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        # Blend with original image
        result = cv2.addWeighted(image, 0.7, enhanced_bgr, 0.3, 0)

        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)

        return result

    except Exception as e:
        print(f"Error in enhance_crosswalk: {str(e)}")
        return image


def preprocess_dataset(input_dir, output_dir):
    """
    Preprocess all images in a directory to enhance traffic signal colors.

    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save enhanced images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    processed_count = 0
    error_count = 0

    for img_path in input_dir.glob("*.[jJ][pP][gG]"):
        try:
            # Read image
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"Warning: Could not read {img_path}")
                error_count += 1
                continue

            # Enhance colors and crosswalks
            enhanced = enhance_traffic_colors(img)
            enhanced = enhance_crosswalk(enhanced)

            # Save enhanced image
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), enhanced)
            processed_count += 1
            print(f"Processed: {img_path.name}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            error_count += 1
            continue

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhance traffic signal colors in images"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input images directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for enhanced images"
    )

    args = parser.parse_args()
    preprocess_dataset(args.input, args.output)
