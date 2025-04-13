# BOSCH BOROSA - 2025 | Team Tarzan

A comprehensive system that combines YOLOv8-based traffic signal detection with ESP32-powered hardware control for autonomous traffic management.

## Features

- Real-time traffic signal and crosswalk detection using YOLOv8
- MQTT-based communication between detection system and hardware
- ESP32-powered traffic pod with autonomous control
- Multiple operational states (GO, SLOW, HALT, GO_SLOW, SLOW_GO)
- Buzzer feedback system for safety alerts
- Roboflow integration for enhanced detection capabilities

## Project Structure

```
.
├── src/                    # Source code directory
├── hardware/              # Hardware-related files
│   └── esp32_traffic_pod.ino  # ESP32 firmware
├── detections/            # Detection results and models
├── DataSet/              # Training dataset
├── Traffic.v3i.yolov8/   # YOLOv8 model configuration
├── main.py               # Main detection script
├── traffic_pod_controller.py  # MQTT controller
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Hardware Components

- ESP32 microcontroller
- Motor drivers
- Buzzer for alerts
- LED indicators
- Power supply system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kushagra-a-singh/BOSCH-BOROSA-Team-Tarzan.git
cd borosa
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following:
```
ROBOFLOW_API_KEY=your_api_key
MQTT_BROKER=your_broker_address
MQTT_PORT=your_broker_port
```

4. Upload ESP32 firmware:
- Open `hardware/esp32_traffic_pod.ino` in Arduino IDE
- Install required libraries
- Upload to ESP32

## Usage

### Running the Detection System

```bash
python main.py
```

The system will:
- Initialize the YOLOv8 model
- Connect to MQTT broker
- Start processing video feed
- Send detection results to traffic pod controller

### Running the Traffic Pod Controller

```bash
python traffic_pod_controller.py
```

The controller will:
- Connect to MQTT broker
- Process detection results
- Control the traffic pod based on signals
- Manage buzzer alerts and state transitions

## System States

- **GO**: Normal forward movement
- **SLOW**: Reduced speed movement
- **HALT**: Complete stop
- **GO_SLOW**: Right turn
- **SLOW_GO**: Left turn

## Requirements

### Software
- Python 3.8+
- PyTorch >= 2.0.0
- Ultralytics YOLOv8 >= 8.3.0
- OpenCV >= 4.8.0
- Paho MQTT Client
- Other dependencies listed in requirements.txt

### Hardware
- ESP32 Development Board
- Motor Drivers
- Buzzer
- LEDs
- Power Supply (5V/12V)
