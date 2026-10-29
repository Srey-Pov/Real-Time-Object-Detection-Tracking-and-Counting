from pathlib import Path
from ultralytics import YOLO
import cv2
import time

# ------------------------------------
# 1. CONFIGURATION
# ------------------------------------
CONFIDENCE = 0.50
LINE_POSITION = 0.80

# Fixed display dimensions
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Reference dimensions for overlay sizing
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720
# ------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------
def get_display_scale(frame):
    """Calculate scale relative to the reference dimensions."""
    h, w = frame.shape[:2]

    return min(
        w / REFERENCE_WIDTH,
        h / REFERENCE_HEIGHT
    )


def draw_scaled_text(
    frame,
    text,
    position,
    font_scale=0.9,
    color=(255, 255, 255),
    thickness=2
):
    """Draw text scaled according to the original frame size."""
    scale = get_display_scale(frame)

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.3, font_scale * scale),
        color,
        max(1, round(thickness * scale)),
        cv2.LINE_AA
    )


def resize_for_display(frame):
    """Fit a frame into a fixed-size canvas without stretching."""
    h, w = frame.shape[:2]

    scale = min(
        DISPLAY_WIDTH / w,
        DISPLAY_HEIGHT / h
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=interpolation
    )

    # Center the resized frame on a fixed-size canvas
    canvas = cv2.copyMakeBorder(
        resized,
        (DISPLAY_HEIGHT - new_h) // 2,
        DISPLAY_HEIGHT - new_h
        - (DISPLAY_HEIGHT - new_h) // 2,
        (DISPLAY_WIDTH - new_w) // 2,
        DISPLAY_WIDTH - new_w
        - (DISPLAY_WIDTH - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return canvas


# ------------------------------------
# 3. PROJECT PATHS
# ------------------------------------
root = Path(__file__).resolve().parents[1]

model_path = root / "models" / "yolov12s_best.pt"

video_candidates = [
    root / "data" / "videos" / "can.mp4",
]

video_path = next(
    (p for p in video_candidates if p.exists()),
    None
)

if video_path is None:
    available_videos = sorted(
        (root / "data" / "videos").glob("*.mp4")
    )

    if not available_videos:
        raise FileNotFoundError(
            f"No video file found in: {root / 'data' / 'videos'}"
        )

    video_path = available_videos[0]

VIDEO_NAME = video_path.name

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found at: {model_path}"
    )


# ------------------------------------
# 4. OUTPUT PATHS
# ------------------------------------
output_dir = root / "output" / "counting"
output_dir.mkdir(parents=True, exist_ok=True)

video_output = output_dir / f"counted_{VIDEO_NAME}"
summary_path = output_dir / f"summary_{video_path.stem}.txt"


# ------------------------------------
# 5. LOAD YOLO MODEL
# ------------------------------------
print("\nLoading model...")

model = YOLO(str(model_path))

print("Model:", model_path)
print("Classes:", model.names)
print("Confidence:", CONFIDENCE)


# ------------------------------------
# 6. OPEN VIDEO
# ------------------------------------
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {video_path}"
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    fps_video = 20.0

total_frames_video = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"Video: {VIDEO_NAME}")
print(f"Resolution: {width}x{height}")
print(f"Video FPS: {fps_video:.2f}")
print(f"Total frames: {total_frames_video}")


# ------------------------------------
# 7. CREATE OUTPUT VIDEO WRITER
# ------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(video_output),
    fourcc,
    fps_video,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not create output video."
    )


# ------------------------------------
# 8. COUNTING SETTINGS
# ------------------------------------
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


# ------------------------------------
# 9. FPS SETTINGS
# ------------------------------------
prev_time = time.time()
fps_display = 0.0
frame_number = 0


# ------------------------------------
# 10. CREATE DISPLAY WINDOW
# ------------------------------------
window_name = "Object Counting"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)


# ------------------------------------
# 11. MAIN PROCESSING LOOP
# ------------------------------------
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Overlay scale is based on source frame dimensions
        scale = get_display_scale(frame)

        # --------------------------------------
        # YOLO DETECTION AND BYTETRACK
        # --------------------------------------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False,
        )

        boxes = results[0].boxes

        # --------------------------------------
        # DRAW COUNTING LINE
        # --------------------------------------
        cv2.line(
            frame,
            (line_x, 0),
            (line_x, height),
            (0, 0, 255),
            max(1, round(3 * scale))
        )

        draw_scaled_text(
            frame,
            "COUNTING LINE",
            (
                max(line_x - int(150 * scale), 10),
                max(int(30 * scale), 15)
            ),
            font_scale=0.6,
            color=(0, 0, 255),
            thickness=2
        )

        # --------------------------------------
        # DETECTION AND OBJECT COUNTING
        # --------------------------------------
        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):
                cls = int(box.cls[0])
                class_name = model.names[cls].lower()

                if class_name not in class_counts:
                    continue

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                color = colors[class_name]

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    max(1, round(2 * scale))
                )

                # Draw tracking label
                label = (
                    f"{class_name} ID:{track_id} {conf:.2f}"
                )

                draw_scaled_text(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - int(10 * scale), int(20 * scale))
                    ),
                    font_scale=0.55,
                    color=color,
                    thickness=2
                )

                # Draw object center point
                cv2.circle(
                    frame,
                    (cx, cy),
                    max(2, round(5 * scale)),
                    (0, 255, 255),
                    -1
                )

                # ----------------------------------
                # CHECK LINE CROSSING
                # ----------------------------------
                previous_x = track_history.get(
                    track_id,
                    cx
                )

                crossed_left_to_right = (
                    previous_x < line_x
                    and cx >= line_x
                )

                crossed_right_to_left = (
                    previous_x > line_x
                    and cx <= line_x
                )

                crossed_line = (
                    crossed_left_to_right
                    or crossed_right_to_left
                )

                # Count each track ID only once
                if (
                    crossed_line
                    and track_id not in counted_ids
                ):
                    counted_ids.add(track_id)
                    class_counts[class_name] += 1

                    print(
                        f"COUNTED -> {class_name} "
                        f"ID:{track_id} "
                        f"Count:{class_counts[class_name]}"
                    )

                track_history[track_id] = cx

        # --------------------------------------
        # CALCULATE PROCESSING FPS
        # --------------------------------------
        current_time = time.time()
        time_difference = current_time - prev_time

        if time_difference > 0:
            fps_display = 1.0 / time_difference

        prev_time = current_time

        total_count = (
            class_counts["can"]
            + class_counts["plastic bottle"]
        )

        # --------------------------------------
        # DRAW COUNT INFORMATION
        # --------------------------------------
        draw_scaled_text(
            frame,
            f"Can: {class_counts['can']}",
            (int(20 * scale), int(45 * scale)),
            font_scale=0.9,
            color=colors["can"],
            thickness=2
        )

        draw_scaled_text(
            frame,
            f"Bottle: {class_counts['plastic bottle']}",
            (int(20 * scale), int(85 * scale)),
            font_scale=0.9,
            color=colors["plastic bottle"],
            thickness=2
        )

        draw_scaled_text(
            frame,
            f"Total: {total_count}",
            (int(20 * scale), int(125 * scale)),
            font_scale=0.9,
            color=(0, 255, 255),
            thickness=2
        )

        draw_scaled_text(
            frame,
            f"FPS: {fps_display:.1f}",
            (int(20 * scale), int(165 * scale)),
            font_scale=0.8,
            color=(255, 255, 255),
            thickness=2
        )

        draw_scaled_text(
            frame,
            f"Frame: {frame_number}/{total_frames_video}",
            (int(20 * scale), int(205 * scale)),
            font_scale=0.7,
            color=(255, 255, 255),
            thickness=2
        )

        # --------------------------------------
        # SAVE ANNOTATED FRAME AT SOURCE SIZE
        # --------------------------------------
        writer.write(frame)

        # --------------------------------------
        # RESIZE ONLY FOR DISPLAY
        # --------------------------------------
        display_frame = resize_for_display(frame)

        cv2.imshow(
            window_name,
            display_frame
        )

        # Press ESC to stop
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ------------------------------------
# 12. SAVE COUNTING SUMMARY
# ------------------------------------
total_count = (
    class_counts["can"]
    + class_counts["plastic bottle"]
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    f.write("----------------------------\n")
    f.write("OBJECT COUNTING SUMMARY\n")
    f.write("----------------------------\n\n")

    f.write(f"Video: {VIDEO_NAME}\n")
    f.write(f"Model: {model_path.name}\n")
    f.write("Tracker: ByteTrack\n")
    f.write(f"Confidence: {CONFIDENCE}\n")
    f.write(f"Video FPS: {fps_video:.2f}\n")
    f.write(f"Resolution: {width}x{height}\n")
    f.write(
        f"Frames Processed: {frame_number}\n\n"
    )

    f.write(f"Can Count: {class_counts['can']}\n")
    f.write(
        f"Plastic Bottle Count: "
        f"{class_counts['plastic bottle']}\n"
    )
    f.write(f"Total Count: {total_count}\n")

print("\n----------------------------")
print("OBJECT COUNTING FINISHED")
print("----------------------------")
print(f"Can: {class_counts['can']}")
print(f"Plastic Bottle: {class_counts['plastic bottle']}")
print(f"Total: {total_count}")
print(f"\nOutput video:\n{video_output}")
print(f"\nSummary:\n{summary_path}")
