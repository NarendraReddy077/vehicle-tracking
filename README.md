# Vehicle Tracking & Counting System

A real-time vehicle detection, tracking, and counting application built with **YOLO**, **BoT-SORT**, and **OpenCV**.

---

## Features

- **Multi-Class Tracking**: Supports detection and tracking for cars, motorcycles, buses, and trucks.
- **Interactive CLI**: Choose specific vehicle categories to track during runtime.
- **Line-Crossing Counter**: Tracks trajectory centroids and counts vehicles crossing a predefined boundary line.
- **Live Visuals & Overlay**: Real-time bounding boxes, ID badges, and on-screen tally display.
- **Video Export & Logging**: Outputs annotated video to `output_videos/` and maintains rotating logs in `logs/`.

---

## Project Structure

```text
vehicle-tracking/
├── input_videos/       # Input video files (e.g. traffic.mp4)
├── models/             # YOLO model weights (e.g. yolo26n.pt)
├── output_videos/      # Processed annotated videos
├── logs/               # Application logs (tracking.log)
├── src/
│   ├── main.py         # Main tracking & counting pipeline
│   └── logger.py       # Rotating file & console logger
├── pyproject.toml      # Project configuration & dependencies
└── README.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/NarendraReddy077/vehicle-tracking.git
cd vehicle-tracking
```

### 2. Install dependencies
Using **uv** (recommended):
```bash
uv sync
```
Or using **pip**:
```bash
pip install ultralytics opencv-python
```

---

## Usage

1. Place your input video inside `input_videos/` (e.g., `input_videos/traffic.mp4`).
2. Ensure the YOLO model weight file exists in `models/` (e.g., `models/yolo26n.pt`).
3. Run the script:
   ```bash
   python src/main.py
   ```
4. Enter the vehicle types you want to track when prompted (e.g., `car, motorcycle` or `car, bus, truck`).
5. Press **`q`** in the video window at any time to stop processing.
