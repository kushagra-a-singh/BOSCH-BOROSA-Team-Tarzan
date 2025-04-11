import json
import time
from enum import Enum

import paho.mqtt.client as mqtt


# Define states as an enumeration
class PodState(Enum):
    GO = "go"
    SLOW = "slow"
    HALT = "halt"
    GO_SLOW = "go_slow"
    SLOW_GO = "slow_go"


class TrafficPodController:
    def __init__(self, broker_address="localhost", broker_port=1883):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.current_state = PodState.SLOW
        self.buzzer_active = False
        self.last_crosswalk_time = 0

    def connect(self):
        try:
            self.client.connect(self.broker_address, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Connection failed: {e}")

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        # Subscribe to relevant topics
        self.client.subscribe("traffic/detection")
        self.client.subscribe("traffic/control")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == "traffic/detection":
                self.handle_detection(payload)
            elif msg.topic == "traffic/control":
                self.handle_control(payload)
        except Exception as e:
            print(f"Error processing message: {e}")

    def handle_detection(self, detection):
        """Handle detection results from YOLOv8 model"""
        crosswalk_detected = detection.get("crosswalk", False)
        green_signal = detection.get("green", False)
        red_signal = detection.get("red", False)

        if crosswalk_detected:
            self.handle_crosswalk(green_signal, red_signal)
        elif red_signal:
            self.transition_to_state(PodState.SLOW)
        elif not (green_signal or red_signal):
            self.transition_to_state(PodState.SLOW)

    def handle_crosswalk(self, green_signal, red_signal):
        """Handle behavior at crosswalk"""
        current_time = time.time()

        # Always halt at crosswalk first
        self.transition_to_state(PodState.HALT)

        if green_signal:
            # Beep for 2 seconds at green signal
            self.activate_buzzer(duration=2)
            time.sleep(2)
            self.transition_to_state(PodState.SLOW_GO)
        elif red_signal:
            # Beep for 5 seconds at red signal
            self.activate_buzzer(duration=5)
            self.last_crosswalk_time = current_time

    def handle_control(self, control):
        """Handle manual control inputs"""
        if "state" in control:
            try:
                new_state = PodState(control["state"])
                self.transition_to_state(new_state)
            except ValueError:
                print(f"Invalid state received: {control['state']}")

    def transition_to_state(self, new_state):
        """Handle state transitions and publish to ESP32"""
        self.current_state = new_state

        # Prepare state data for ESP32
        state_data = {"state": new_state.value, "buzzer": self.buzzer_active}

        # Add speed values based on state
        if new_state == PodState.GO_SLOW:
            state_data["speed_left"] = 100
            state_data["speed_right"] = 0
        elif new_state == PodState.SLOW_GO:
            state_data["speed_left"] = 0
            state_data["speed_right"] = 100

        # Publish state to ESP32
        self.client.publish("traffic/esp32/state", json.dumps(state_data))

    def activate_buzzer(self, duration=None):
        """Control buzzer activation"""
        self.buzzer_active = True
        if duration:
            time.sleep(duration)
            self.buzzer_active = False

    def run(self):
        """Main loop"""
        try:
            while True:
                # Check if we need continuous buzzer after red light violation
                if (
                    self.current_state in [PodState.GO, PodState.SLOW_GO]
                    and self.last_crosswalk_time > 0
                ):
                    self.buzzer_active = True
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    controller = TrafficPodController()
    controller.connect()
    controller.run()
