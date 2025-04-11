import time

import cv2


def main():
    # Initialize video capture
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera initialized successfully")
    print("Press 'q' to quit")

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Display the frame
            cv2.imshow("Camera Test", frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
