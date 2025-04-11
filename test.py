import base64
import json
import os
import threading
import time
import winsound  # For buzzer sound
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

# Configuration
SAVE_DETECTIONS = os.getenv("SAVE_DETECTIONS", "false").lower() == "true"
DETECTION_DIR = "detections"
TEMP_DIR = "temp"

# Buzzer configuration
BUZZER_FREQUENCY = 1000  # Hz
BUZZER_DURATION_SELECTED = 2000  # 2 seconds in milliseconds
BUZZER_DURATION_DETECTED = 5000  # 5 seconds in milliseconds

# Create directories if saving is enabled
if SAVE_DETECTIONS:
    os.makedirs(DETECTION_DIR, exist_ok=True)
    # Create subdirectories for each class
    for class_name in ["red", "green", "crosswalk", "no", "multiple"]:
        os.makedirs(os.path.join(DETECTION_DIR, class_name), exist_ok=True)

# Create temp directory
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize the Roboflow client
client = InferenceHTTPClient(
    api_url=os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com"),
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

# Communication Configuration
ENABLE_MQTT = os.getenv("ENABLE_MQTT", "false").lower() == "true"
ENABLE_HTTP = os.getenv("ENABLE_HTTP", "false").lower() == "true"

mqtt_client = None
if ENABLE_MQTT:
    try:
        import paho.mqtt.client as mqtt

        MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
        MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
        MQTT_TOPIC = "traffic/detection"

        mqtt_client = mqtt.Client()
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("MQTT connection successful")
    except Exception as e:
        print(f"MQTT setup skipped: {e}")
        mqtt_client = None


def play_buzzer(duration_ms):
    """Play buzzer sound for specified duration in a separate thread"""

    def buzzer_thread():
        winsound.Beep(BUZZER_FREQUENCY, duration_ms)

    # Start buzzer in a separate thread to avoid blocking
    threading.Thread(target=buzzer_thread).start()


def send_control_commands(action, buzzer):
    """Send control commands via enabled communication channels"""
    # Prepare control message for MQTT
    control_message = {"action": action, "buzzer": buzzer, "timestamp": time.time()}

    # Play buzzer for 2 seconds when action is selected
    if action == "GO":
        play_buzzer(BUZZER_DURATION_SELECTED)

    # Send via MQTT if enabled and available
    if ENABLE_MQTT and mqtt_client:
        try:
            mqtt_client.publish(MQTT_TOPIC, json.dumps(control_message))
            print("[MQTT] Command sent successfully")
        except Exception as e:
            print(f"[MQTT] Send failed: {e}")

    # Send movement commands via HTTP
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        base_url = "http://192.168.125.246"

        # Complete endpoint mapping for all movements
        if action == "GO":
            endpoint = "/backward"  # When 'no' is detected
        elif action == "FORWARD":
            endpoint = "/forward"  # When explicitly requested to go forward
        else:  # SLOW or any other case
            endpoint = "/stop"

        movement_url = f"{base_url}{endpoint}"
        max_retries = 3
        current_retry = 0
        success = False

        while current_retry < max_retries and not success:
            try:
                current_retry += 1
                print(f"[Movement] Attempt {current_retry}/{max_retries}")
                response = session.get(movement_url, timeout=5)
                print(
                    f"[Movement] Command sent to {movement_url}, status: {response.status_code}"
                )

                if response.status_code in [200, 201, 202]:
                    success = True
                    break
                else:
                    print(
                        f"[Movement] Warning: Unexpected status code {response.status_code}"
                    )
                    if current_retry < max_retries:
                        print(f"[Movement] Retrying in 1 second...")
                        time.sleep(1)

            except requests.exceptions.Timeout:
                print(f"[Movement] Attempt {current_retry} timed out.")
                if current_retry < max_retries:
                    print(f"[Movement] Retrying in 1 second...")
                    time.sleep(1)
            except requests.exceptions.ConnectionError:
                print(f"[Movement] Attempt {current_retry} connection failed.")
                if current_retry < max_retries:
                    print(f"[Movement] Retrying in 1 second...")
                    time.sleep(1)
            except Exception as e:
                print(f"[Movement] Attempt {current_retry} failed: {str(e)}")
                if current_retry < max_retries:
                    print(f"[Movement] Retrying in 1 second...")
                    time.sleep(1)

        if not success:
            print("[Movement] All attempts failed.")

        session.close()

        # Send additional control commands if HTTP control is enabled
        if ENABLE_HTTP:
            try:
                POD_API_URL = os.getenv("POD_API_URL", "http://localhost:5000/control")
                response = requests.post(POD_API_URL, json=control_message, timeout=3)
                print(f"[HTTP] Control command sent, status: {response.status_code}")
            except Exception as e:
                print(f"[HTTP] Control command failed: {e}")

    except ImportError:
        print("[Movement] Requests library not available for HTTP commands")


def process_detection(predictions):
    """Process detections and determine action"""
    if not predictions:
        return "SLOW", False  # Default to SLOW (stop) for safety

    # Initialize variables
    detected_green = False
    detected_red = False
    highest_confidence = 0
    action = "SLOW"  # Default to SLOW (stop)
    buzzer = False

    # Process each detection
    for pred in predictions:
        class_name = pred["class"].lower()
        confidence = pred["confidence"]

        print(
            f"[DEBUG] Detected {class_name} with confidence {confidence}"
        )  # Debug line

        if confidence > highest_confidence:
            highest_confidence = confidence

        # Check for green signal with lower confidence threshold
        if class_name == "green" and confidence > 0.10:  # Using 0.10 threshold
            detected_green = True
            print("[DEBUG] Detected green signal")
            # Play buzzer for 2 seconds when green is detected
            play_buzzer(BUZZER_DURATION_SELECTED)

        # Check for red signal
        if class_name == "red" and confidence > 0.10:
            detected_red = True
            print("[DEBUG] Detected red signal")
            # Play buzzer for 5 seconds when red is detected
            play_buzzer(BUZZER_DURATION_DETECTED)

    # Simplified decision logic - GO on green, SLOW otherwise
    if detected_green:
        action = "GO"  # This will trigger backward movement in send_control_commands
        buzzer = True
        print("[DEBUG] Setting action to GO due to green signal")
    else:
        action = "SLOW"
        buzzer = False

    print(f"[DEBUG] Final action decided: {action}")
    return action, buzzer


def save_detection_frame(frame, predictions, action):
    """Save frame with detections if enabled"""
    if not SAVE_DETECTIONS or not predictions:
        return

    # Determine directory based on detections
    if len(predictions) > 1:
        save_dir = "multiple"
    else:
        save_dir = predictions[0]["class"].lower()

    # Create filename with timestamp and action
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confidence = max(pred["confidence"] for pred in predictions)
    filename = f"{timestamp}_{action}_{confidence:.2f}.jpg"

    # Full path for saving
    save_path = os.path.join(DETECTION_DIR, save_dir, filename)

    # Save the frame
    cv2.imwrite(save_path, frame)
    print(f"[SAVE] Frame saved: {save_path}")


def main():
    print("\n=== Traffic Detection System ===")
    print("System Configuration:")
    print(f"- MQTT Enabled: {ENABLE_MQTT}")
    print(f"- HTTP Enabled: {ENABLE_HTTP}")
    print(f"- Save Detections: {SAVE_DETECTIONS}")
    if SAVE_DETECTIONS:
        print(f"- Save Directory: {os.path.abspath(DETECTION_DIR)}")
    print("\nInitializing camera stream...")

    # IP Camera stream URL
    stream_url = "http://192.168.125.246:81/stream"
    print(f"Connecting to stream: {stream_url}")

    def init_camera():
        try:
            # First try to ping the device
            import requests

            response = requests.get(f"http://192.168.125.246/status", timeout=2)
            if response.status_code != 200:
                print("Warning: Device status check failed")
        except:
            print("Warning: Could not check device status")

        cap = cv2.VideoCapture(
            stream_url, cv2.CAP_FFMPEG
        )  # Explicitly use FFMPEG backend
        if cap.isOpened():
            # Set OpenCV capture properties for better streaming
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increased buffer size slightly
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 10)  # Reduced FPS further for stability
            # Set timeouts
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout
        return cap

    # Initialize video capture with IP camera stream
    cap = init_camera()
    reconnect_delay = 2  # Initial delay between reconnection attempts
    max_reconnect_delay = 30  # Maximum delay between attempts
    reconnect_attempts = 0
    max_reconnect_attempts = 10

    if not cap.isOpened():
        print("Error: Could not connect to IP camera stream")
        print("Please check:")
        print("1. Device is powered on")
        print("2. Device IP address is correct (192.168.125.246)")
        print("3. Device is connected to the network")
        print("4. No firewall is blocking the connection")
        return

    print("Camera stream initialized successfully")
    print("Press 'q' to quit")

    last_process_time = 0
    process_interval = 5.0  # Changed from 3 to 5 seconds
    is_processing = False
    last_action = "SLOW"  # Default action
    last_predictions = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(
                    f"\nError: Could not read frame from stream (Attempt {reconnect_attempts + 1}/{max_reconnect_attempts})"
                )
                reconnect_attempts += 1

                if reconnect_attempts >= max_reconnect_attempts:
                    print("Max reconnection attempts reached. Exiting...")
                    break

                print(f"Waiting {reconnect_delay} seconds before reconnecting...")
                time.sleep(reconnect_delay)

                # Increase delay for next attempt (exponential backoff)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

                # Close current connection
                cap.release()
                cv2.destroyAllWindows()

                print("Attempting to reconnect to stream...")
                cap = init_camera()

                if cap.isOpened():
                    print("Successfully reconnected to stream")
                    reconnect_attempts = 0
                    reconnect_delay = 2  # Reset delay on successful connection
                continue

            # Reset reconnect attempts and delay on successful frame read
            if reconnect_attempts > 0:
                print("Stream connection restored")
                reconnect_attempts = 0
                reconnect_delay = 2

            current_time = time.time()
            frame_display = frame.copy()

            # Draw last known detections
            for pred in last_predictions:
                x = pred["x"]
                y = pred["y"]
                width = pred["width"]
                height = pred["height"]

                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                # Draw rectangle with color based on class
                color = (0, 255, 0)  # Default green
                if pred["class"].lower() == "red":
                    color = (0, 0, 255)  # Red
                elif pred["class"].lower() == "crosswalk":
                    color = (255, 0, 0)  # Blue

                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)

                # Add label
                label = f"{pred['class']} {pred['confidence']:.2f}"
                cv2.putText(
                    frame_display,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Add status overlay
            status_text = f"Status: {'Processing' if is_processing else 'Monitoring'}"
            action_text = f"Action: {last_action}"
            time_to_next = max(0, process_interval - (current_time - last_process_time))
            timer_text = f"Next update in: {time_to_next:.1f}s"
            stream_text = "Stream: Connected"

            # Draw status information
            cv2.putText(
                frame_display,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame_display,
                action_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame_display,
                timer_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame_display,
                stream_text,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Process frame every interval
            if current_time - last_process_time >= process_interval:
                is_processing = True
                print("\n--- Processing frame ---")

                # Save frame temporarily
                temp_path = os.path.join(TEMP_DIR, "temp_frame.jpg")
                cv2.imwrite(temp_path, frame)

                try:
                    # Get predictions from Roboflow
                    result = client.run_workflow(
                        workspace_name="borosa",
                        workflow_id="detect-count-and-visualize",
                        images={"image": temp_path},
                        use_cache=True,
                    )

                    if result and len(result) > 0:
                        predictions = (
                            result[0].get("predictions", {}).get("predictions", [])
                        )
                        last_predictions = predictions

                        # Process detections and get action
                        action, buzzer = process_detection(predictions)
                        last_action = action

                        # Save frame if there are detections
                        if predictions:
                            save_detection_frame(frame_display, predictions, action)

                        # Send control commands
                        send_control_commands(action, buzzer)

                        # Display results
                        print(f"Detection Results:")
                        print(f"- Action: {action}")
                        print(f"- Buzzer: {buzzer}")
                        print(f"- Objects detected: {len(predictions)}")
                        for pred in predictions:
                            print(f"  * {pred['class']}: {pred['confidence']:.2f}")

                except Exception as e:
                    print(f"Error processing frame: {e}")

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    is_processing = False
                    last_process_time = current_time

            # Display the frame
            cv2.imshow("Traffic Detection", frame_display)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuitting...")
                break

    except Exception as e:
        print(f"\nError in main loop: {e}")

    finally:
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
