from pathlib import Path
from ultralytics import YOLO
import cv2

# Build project paths safely
root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "yolov12s_best.pt"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at: {model_path}")

# Load trained YOLO model
model = YOLO(str(model_path))

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Please connect a camera and try again.")

print("Webcam opened. Press ESC to exit.")

# Real-time detection
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read frame from webcam.")
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Run YOLO detection
    results = model(
        frame,
        conf=0.70,
        iou=0.5,
        verbose=False
    )

    # Draw detection results
    annotated_frame = results[0].plot()

    # Show webcam
    cv2.imshow("Object Detection - Can & Plastic Bottle", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()