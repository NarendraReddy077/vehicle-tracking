import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "models/yolo26n.pt"
VIDEO_PATH = "input_videos/traffic.mp4"
OUTPUT_PATH = "output_videos/output_tracked.mp4"

LINE_X = 500
CONFIDENCE = 0.4

# COCO vehicle classes
VEHICLE_CLASSES = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}


print("\n========== VEHICLE TRACKING ==========")
print("Available vehicles:")
for vehicle in VEHICLE_CLASSES.keys():
    print(f"  {vehicle}")