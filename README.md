# Traffic Signal Detection System

A YOLOv8-based system for detecting traffic signals and crosswalks, with image enhancement capabilities.

## Features

- Traffic signal and crosswalk detection using YOLOv8
- Color enhancement for better detection accuracy
- Interactive Streamlit web interface for testing
- Comprehensive accuracy metrics and validation reports
- Support for both CPU and GPU training

## Project Structure

```
.
├── src/
│   ├── main.py           # Training script with validation
│   ├── app.py            # Streamlit web interface
│   └── color_enhance.py  # Image enhancement utilities
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-signal-detection.git
cd traffic-signal-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model:

```bash
python src/main.py
```

The training script will:
- Verify and enhance the dataset
- Train the YOLOv8 model
- Print detailed accuracy metrics
- Save the best model weights

### Testing with Web Interface

To use the interactive web interface:

```bash
streamlit run src/app.py
```

Features:
- Upload and test images
- Select from available trained models
- Adjust confidence threshold
- Toggle image enhancement options
- View detection results in real-time

## Model Performance

The system provides detailed performance metrics including:
- Overall accuracy
- Per-class accuracy
- Precision and recall
- Confusion matrix
- Confidence score statistics

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- Streamlit
- Other dependencies listed in requirements.txt

## License

[Your License]

## Contributing

[Your Contributing Guidelines] 