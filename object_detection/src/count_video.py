from pathlib import Path
from ultralytics import YOLO
import cv2
import time

# Configuration
CONFIDENCE = 0.50
LINE_POSITION = 0.80

# Paths
root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "yolo11s_best.pt"
video_candidates = [
    root / "data" / "videos" / "fast.mp4",
    root / "data" / "videos" / "*.mp4",
]

video_path = next(
    (p for p in video_candidates if isinstance(p, Path) and p.exists()),
    None
)

if video_path is None:
    available_videos = sorted((root / "data" / "videos").glob("*.mp4"))
    if not available_videos:
        raise FileNotFoundError(
            f"No video file found in: {root / 'data' / 'videos'}"
        )
    video_path = available_videos[0]

VIDEO_NAME = video_path.name

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at: {model_path}")

# Output paths
output_dir = root / "output" / "counting"
output_dir.mkdir(parents=True, exist_ok=True)
video_output = output_dir / f"counted_{VIDEO_NAME}"
summary_path = output_dir / "summary.txt"

# Load model
print("\nLoading model...")
model = YOLO(str(model_path))
print("Model:", model_path)
print("Classes:", model.names)
print("Confidence:", CONFIDENCE)

# Open video
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
if fps_video <= 0:
    fps_video = 20

total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(video_output), fourcc, fps_video, (width, height))
if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

# Counting settings
line_x = int(width * LINE_POSITION)
class_counts = {
    "can": 0,
    "plastic bottle": 0,
}
counted_ids = set()
track_history = {}
colors = {
    "can": (255, 0, 0),
    "plastic bottle": (0, 255, 0),
}

# FPS
prev_time = time.time()
fps_display = 0.0
frame_number = 0

cv2.namedWindow("Object Counting", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        verbose=False,
    )
    boxes = results[0].boxes

    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)
    cv2.putText(frame, "COUNTING LINE", (line_x - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, ids):
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()

            if class_name not in class_counts:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            color = colors.get(class_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ID:{track_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            previous_x = track_history.get(track_id, cx)
            crossed_left_to_right = previous_x < line_x and cx >= line_x
            crossed_right_to_left = previous_x > line_x and cx <= line_x
            crossed_line = crossed_left_to_right or crossed_right_to_left

            if crossed_line and track_id not in counted_ids:
                counted_ids.add(track_id)
                class_counts[class_name] += 1
                print(f"COUNTED -> {class_name} ID:{track_id} Total:{class_counts[class_name]}")

            track_history[track_id] = cx

    current_time = time.time()
    time_difference = current_time - prev_time
    if time_difference > 0:
        fps_display = 1 / time_difference
    prev_time = current_time

    total_count = class_counts["can"] + class_counts["plastic bottle"]

    cv2.putText(frame, f"Can: {class_counts['can']}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors["can"], 2)
    cv2.putText(frame, f"Bottle: {class_counts['plastic bottle']}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors["plastic bottle"], 2)
    cv2.putText(frame, f"Total: {total_count}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_number}/{total_frames_video}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    writer.write(frame)
    cv2.imshow("Object Counting", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# Save summary
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("----------------------------\n")
    f.write("OBJECT COUNTING SUMMARY\n")
    f.write("----------------------------\n\n")
    f.write(f"Video: {VIDEO_NAME}\n")
    f.write("Model: YOLO11s + ByteTrack\n")
    f.write(f"Confidence: {CONFIDENCE}\n")
    f.write(f"Video FPS: {fps_video:.2f}\n")
    f.write(f"Frames Processed: {frame_number}\n\n")
    f.write(f"Can Count: {class_counts['can']}\n")
    f.write(f"Plastic Bottle Count: {class_counts['plastic bottle']}\n")
    f.write(f"Total Count: {total_count}\n")

print("\n---------------------------")
print("OBJECT COUNTING FINISHED")
print("----------------------------")
print(f"Can: {class_counts['can']}")
print(f"Plastic Bottle: {class_counts['plastic bottle']}")
print(f"Total: {total_count}")
print(f"\nOutput video:\n{video_output}")
print(f"\nSummary:\n{summary_path}")