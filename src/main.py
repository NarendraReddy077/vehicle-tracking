import os
import sys
import cv2
from ultralytics import YOLO
from logger import get_logger

logger = get_logger("vehicle_tracking")
logger.info("Initializing Vehicle Tracking System...")

# Configuration
MODEL_PATH = "models/yolo26n.pt"
VIDEO_PATH = "input_videos/traffic2.mp4"
OUTPUT_PATH = "output_videos/output_tracked.mp4"

LINE_X = 500
CONFIDENCE = 0.4
MAX_TRACK_AGE = 30
SPATIAL_DEDUPE_DISTANCE = 50
SPATIAL_DEDUPE_FRAMES = 15

# COCO vehicle classes
VEHICLE_CLASSES = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}

print("\n========== VEHICLE TRACKING ==========")
print("Vehicle classes to track:")
for vehicle in VEHICLE_CLASSES.keys():
    print(f"  {vehicle}")

while True:
    try:
        user_input = input(
            "\nWhich vehicles do you want to track? "
            "(e.g. car, truck): "
        ).lower().strip()
    except (EOFError, KeyboardInterrupt):
        logger.warning("Input interrupted by user during vehicle selection. Exiting.")
        sys.exit(1)

    selected_vehicles = list(dict.fromkeys(
        vehicle.strip()
        for vehicle in user_input.split(",")
        if vehicle.strip()
    ))

    if not selected_vehicles:
        logger.warning("No vehicles selected; prompting again.")
        continue

    invalid_vehicles = [
        vehicle
        for vehicle in selected_vehicles
        if vehicle not in VEHICLE_CLASSES
    ]

    if invalid_vehicles:
        error_msg = f"Invalid vehicle(s) selected: {', '.join(invalid_vehicles)}"
        logger.warning(error_msg)
        print("Please choose from:", ", ".join(VEHICLE_CLASSES.keys()))
        continue

    break

selected_class_ids = [
    VEHICLE_CLASSES[vehicle]
    for vehicle in selected_vehicles
]

logger.info(f"Selected vehicles for tracking: {', '.join(selected_vehicles)}")
print("\nTracking:", ", ".join(selected_vehicles))

logger.info(f"Loading YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except FileNotFoundError:
    logger.exception(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)
except Exception:
    logger.exception("Failed to load YOLO model.")
    sys.exit(1)


logger.info(f"Opening video file: {VIDEO_PATH}")
try:
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video opened: {width}x{height} @ {fps:.2f} FPS")

except Exception:
    logger.exception(f"Failed to open video file: {VIDEO_PATH}")
    sys.exit(1)


try:
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for: {OUTPUT_PATH}")

    logger.info(f"Video writer initialized for: {OUTPUT_PATH}")

except Exception:
    logger.exception(f"Failed to initialize video writer for: {OUTPUT_PATH}")
    cap.release()
    sys.exit(1)

track_history = {}  # {track_id: {"x": int, "y": int, "last_seen": int}}
counted_ids = set()
recent_crossings = []  # [{"x": int, "y": int, "frame": int, "vehicle_name": str}]

vehicle_count = {
    vehicle: 0
    for vehicle in selected_vehicles
}

total_crossed = 0
frame_number = 0

logger.info("Starting video tracking stream...")

try:
    while True:

        try:
            success, frame = cap.read()
        except Exception:
            logger.exception(f"Error reading frame {frame_number}. Stopping stream.")
            break

        if not success:
            logger.info("Video stream ended or no more frames to read.")
            break

        frame_number += 1

        result = None
        try:
            results = model.track(
                frame,
                persist=True,
                classes=selected_class_ids,
                tracker="botsort.yaml",
                conf=CONFIDENCE,
                verbose=False
            )
            if results and len(results) > 0:
                result = results[0]
        except Exception:
            logger.exception(f"Tracking inference failed on frame {frame_number}. Processing raw frame.")

        try:
            if result is not None:
                annotated_frame = result.plot(conf=False, line_width=2, font_size=0.4)
            else:
                annotated_frame = frame.copy()
        except Exception:
            logger.exception(f"Failed to annotate frame {frame_number}. Using raw frame instead.")
            annotated_frame = frame.copy()

        cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, height), (0, 255, 0), 3)

        try:
            if result is not None and result.boxes is not None and result.boxes.id is not None:

                track_ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for track_id, box, class_id in zip(track_ids, boxes, class_ids):

                    vehicle_name = next(
                        (name for name, class_id_value in VEHICLE_CLASSES.items()
                        if class_id_value == class_id), None
                    )

                    if vehicle_name is None:
                        continue

                    try:
                        x1, y1, x2, y2 = map(int, box)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        # Line crossing check (survives 1+ frame occlusion via track_history)
                        if track_id in track_history:
                            previous_x = track_history[track_id]["x"]

                            crossed = (previous_x > LINE_X and center_x <= LINE_X)

                            if crossed and track_id not in counted_ids:
                                # Suppress duplicates caused by ID switches near the counting line
                                is_duplicate = any(
                                    rc["vehicle_name"] == vehicle_name and
                                    abs(rc["y"] - center_y) < SPATIAL_DEDUPE_DISTANCE and
                                    (frame_number - rc["frame"]) <= SPATIAL_DEDUPE_FRAMES
                                    for rc in recent_crossings
                                )

                                if not is_duplicate:
                                    total_crossed += 1
                                    vehicle_count[vehicle_name] += 1
                                    counted_ids.add(track_id)
                                    recent_crossings.append({
                                        "x": center_x,
                                        "y": center_y,
                                        "frame": frame_number,
                                        "vehicle_name": vehicle_name
                                    })

                                    logger.info(
                                        f"Vehicle crossed: {vehicle_name.upper()} (ID: {track_id}) | "
                                        f"Category count: {vehicle_count[vehicle_name]} | Total crossed: {total_crossed}"
                                    )
                                else:
                                    logger.info(
                                        f"Duplicate crossing suppressed for ID {track_id} ({vehicle_name}) due to nearby recent count."
                                    )

                        track_history[track_id] = {
                            "x": center_x,
                            "y": center_y,
                            "last_seen": frame_number
                        }

                    except Exception:
                        logger.exception(
                            f"Error processing track ID {track_id} on frame {frame_number}. Skipping this track."
                        )
                        continue

                # Prune tracks older than MAX_TRACK_AGE (maintains history across temporary occlusions)
                track_history = {
                    tid: data
                    for tid, data in track_history.items()
                    if (frame_number - data["last_seen"]) <= MAX_TRACK_AGE
                }

                # Prune expired spatial deduplication history
                recent_crossings = [
                    rc for rc in recent_crossings
                    if (frame_number - rc["frame"]) <= SPATIAL_DEDUPE_FRAMES
                ]

        except Exception:
            logger.exception(f"Error processing detections on frame {frame_number}.")

        try:
            cv2.rectangle(annotated_frame,(10, 10),(280, 60 + len(selected_vehicles) * 30),(0, 0, 0),-1)

            y = 40
            for vehicle in selected_vehicles:
                cv2.putText(
                    annotated_frame,
                    f"{vehicle}: {vehicle_count[vehicle]}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                y += 30

            cv2.putText(
                annotated_frame,
                f"Total: {total_crossed}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
        except Exception:
            logger.exception(f"Failed to draw statistics overlay on frame {frame_number}.")

        try:
            out.write(annotated_frame)
        except Exception:
            logger.exception(f"Failed to write frame {frame_number} to output video.")

        try:
            cv2.imshow("YOLO + BoT-SORT Vehicle Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Video processing stopped by user ('q' key pressed).")
                break
        except Exception:
            logger.exception(f"Display error on frame {frame_number}. Continuing without display.")

except KeyboardInterrupt:
    logger.warning("Video processing interrupted by user (Ctrl+C).")

except Exception:
    logger.exception("Unexpected error occurred during video processing.")

finally:
    try:
        cap.release()
    except Exception:
        logger.exception("Failed to release video capture.")

    try:
        out.release()
    except Exception:
        logger.exception("Failed to release video writer.")

    try:
        cv2.destroyAllWindows()
    except Exception:
        logger.exception("Failed to destroy display windows.")

    logger.info("Resources released and windows closed.")

    logger.info("========== Final Results ==========")
    for vehicle, count in vehicle_count.items():
        logger.info(f"{vehicle.capitalize()}: {count}")

    logger.info(f"Total vehicles crossed: {total_crossed}")
    logger.info(f"Output video saved to: {OUTPUT_PATH}")
    logger.info("Vehicle Tracking process completed successfully.")