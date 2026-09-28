import sys
import cv2
from ultralytics import YOLO
from logger import get_logger

logger = get_logger("vehicle_tracking")
logger.info("Initializing Vehicle Tracking System...")

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

selected_vehicles = []

while True:
    try:
        user_input = input(
            "\nWhich vehicles do you want to track? "
            "(e.g. car, truck): "
        ).lower().strip()
    except (EOFError, KeyboardInterrupt):
        logger.warning("Input interrupted by user during vehicle selection. Exiting.")
        sys.exit(1)

    selected_vehicles = [
        vehicle.strip()
        for vehicle in user_input.split(",")
        if vehicle.strip()
    ]

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

previous_positions = {}
counted_ids = set()

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

        try:
            results = model.track(
                frame,
                persist=True,
                classes=selected_class_ids,
                tracker="botsort.yaml",
                conf=CONFIDENCE,
                verbose=False
            )
            result = results[0]
        except Exception:
            logger.exception(f"Tracking inference failed on frame {frame_number}. Skipping frame.")
            continue

        try:
            annotated_frame = result.plot()
        except Exception:
            logger.exception(f"Failed to annotate frame {frame_number}. Using raw frame instead.")
            annotated_frame = frame.copy()

        cv2.line(annotated_frame,(LINE_X, 0),(LINE_X, height),(0, 255, 0),3)

        try:
            if result.boxes.id is not None:

                track_ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                active_ids = set()

                for track_id, box, class_id in zip(track_ids, boxes, class_ids):

                    active_ids.add(track_id)

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

                        cv2.circle(annotated_frame,(center_x, center_y),5,(0, 0, 255),-1)

                        # Line crossing
                        if track_id in previous_positions:
                            previous_x = previous_positions[track_id]

                            crossed = (previous_x > LINE_X and center_x <= LINE_X)

                            if crossed and track_id not in counted_ids:
                                total_crossed += 1
                                vehicle_count[vehicle_name] += 1
                                counted_ids.add(track_id)

                                logger.info(
                                    f"Vehicle crossed: {vehicle_name.upper()} (ID: {track_id}) | "
                                    f"Category count: {vehicle_count[vehicle_name]} | Total crossed: {total_crossed}"
                                )

                        previous_positions[track_id] = center_x

                        cv2.putText(annotated_frame,f"{vehicle_name} | ID {track_id}",(x1, max(y1 - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 255),2)

                    except Exception:
                        logger.exception(
                            f"Error processing track ID {track_id} on frame {frame_number}. Skipping this track."
                        )
                        continue

                previous_positions = {
                    tid: pos
                    for tid, pos in previous_positions.items()
                    if tid in active_ids
                }

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
        logger.info(f"Total {vehicle.capitalize()}s: {count}")

    logger.info(f"Total vehicles crossed: {total_crossed}")
    logger.info(f"Output video saved to: {OUTPUT_PATH}")
    logger.info("Vehicle Tracking process completed successfully.")