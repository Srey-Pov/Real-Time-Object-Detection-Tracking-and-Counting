# Real-Time Can and Plastic Bottle Detection, Tracking, and Counting

This project detects, tracks, and counts recyclable objects, specifically cans
and plastic bottles, using five YOLO object detection models:

- YOLOv8s
- YOLOv9s
- YOLOv10s
- YOLO11s
- YOLOv12s

The models were trained and evaluated using a common experimental configuration
to compare detection performance and computational efficiency. The project
also includes real-time webcam detection and video-based object tracking and
counting using ByteTrack and virtual line crossing.

## Project Overview

The project consists of two main parts:

1. Comparative evaluation of five YOLO models for can and plastic bottle
   detection.
2. Real-time deployment using YOLO11s, ByteTrack, and virtual line crossing
   for object tracking and counting.

YOLO11s achieved the highest Precision and the lowest GFLOPs among the
evaluated models. YOLO11s was selected for the real-time deployment and
tracking/counting demonstration.

## Project structure

- [requirements.txt](requirements.txt) – Python dependencies
- [object_detection/data](object_detection/data) – training and test images/videos
- [object_detection/models](object_detection/models) – trained model (`best.pt`)
- [object_detection/src/detect_image.py](object_detection/src/detect_image.py) – image detection script
- [object_detection/src/real_time.py](object_detection/src/real_time.py) – webcam detection
- [object_detection/src/count_video.py](object_detection/src/count_video.py) – video counting script
- [object_detection/output](object_detection/output) – saved detection results
- [README.md](README.md) – project documentation

## Setup

1. Open PowerShell in the project root.
2. Create a virtual environment:

```powershell
python -m venv .venv
```

3. If PowerShell blocks script activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

4. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run detection

### Image detection

```powershell
python .\object_detection\src\detect_image.py
```

Use a custom image:

```powershell
python .\object_detection\src\detect_image.py --image .\object_detection\data\images\your_image.jpg --output .\object_detection\output
```

Set a confidence threshold:

```powershell
python .\object_detection\src\detect_image.py
```

### Real-time detection

```powershell
python .\object_detection\src\real_time.py
```

### Video counting

```powershell
python .\object_detection\src\count_video.py
```

## Notes

- The model file is expected at [object_detection/models/best.pt](object_detection/models/best.pt).
- If no image path is provided, the script defaults to [object_detection/data/images/test3.jpg](object_detection/data/images/test3.jpg).
- Output images are stored in [object_detection/output](object_detection/output).
