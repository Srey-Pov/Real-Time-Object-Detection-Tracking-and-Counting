from ultralytics import YOLO
import cv2
import argparse
import os

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Detect waste in image")

    parser.add_argument("--image", help="Path to input image (default: data/images/bottle.jpg)")
    parser.add_argument("--output", default=os.path.join(project_dir, "output"), help="output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    args = parser.parse_args()

    if not args.image:
        args.image = os.path.join(project_dir, "data", "images", "bottle.jpg")

    model_path = os.path.join(project_dir, "models", "yolo11s_best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        print("Please provide a valid image path or add the image to the data/images folder")
        return

    model = YOLO(model_path)
    image = cv2.imread(args.image)

    if image is None:
        print(f"Error: Could not read image {args.image}")
        return

    results = model(image, conf=args.conf)
    annotated_image = results[0].plot()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"detected_{os.path.basename(args.image)}")
    cv2.imwrite(output_path, annotated_image)

    print(f"Detection complete. Saved to {output_path}")

if __name__ == "__main__":
    main()