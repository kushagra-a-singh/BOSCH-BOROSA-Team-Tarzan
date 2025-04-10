from pathlib import Path

import cv2
import numpy as np


def enhance_traffic_colors(image):
    """
    Enhance traffic signal colors in the image.

    Args:
        image: Input image in BGR format

    Returns:
        Enhanced image
    """
    try:
        if image is None:
            raise ValueError("Input image is None")

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split channels
        h, s, v = cv2.split(hsv)

        # Enhance saturation for better color detection
        s = cv2.convertScaleAbs(s, alpha=1.3, beta=0)

        # Apply CLAHE to value channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v = clahe.apply(v)

        # Merge channels
        hsv = cv2.merge([h, s, v])

        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

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

        # Apply CLAHE with stronger contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Apply adaptive thresholding to detect zebra patterns
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )

        # Edge detection with optimized parameters
        edges = cv2.Canny(thresh, 30, 150)

        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        # Blend with original image with adjusted weights
        result = cv2.addWeighted(image, 0.6, enhanced_bgr, 0.4, 0)

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
