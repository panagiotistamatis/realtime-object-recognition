"""
Real-time object recognition demo.

Loads the trained CNN weights and runs a live webcam demo: a center region of
interest (ROI) is classified every frame, the top-3 predictions are shown with
confidence-based colors, predictions are temporally smoothed over several
frames, and an FPS counter is displayed.

Run with:
    python webcam.py

Requires the trained weights file (see README: available under Releases, or
produced by running train.py).
"""
import os
import time
from collections import Counter

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import ImprovedCNN, NUM_CLASSES, CLASS_NAMES, NORM_MEAN, NORM_STD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_FILE = 'simple_classroom_model.pth'


def preprocess_frame(frame):
    """
    Convert a webcam frame into the tensor format expected by the CNN.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def run_webcam_recognition(model, class_names):
    """
    Run real-time object recognition using the webcam.
    """
    print("Starting webcam recognition...")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting real-time recognition. Press 'q' to quit, 's' to save snapshot.")

    # Create directory for saved snapshots
    snapshots_dir = "webcam_snapshots"
    os.makedirs(snapshots_dir, exist_ok=True)

    # Visualization options
    show_fps = True
    show_help = True
    confidence_threshold = 0.3  # Minimum confidence to display a prediction

    # Keep track of prediction history for temporal smoothing
    pred_history = []
    history_len = 5

    # Frame rate calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0

    model.eval()  # Set model to evaluation mode

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Copy frame for display purposes
        display_frame = frame.copy()

        # Draw ROI rectangle in the center
        h, w = frame.shape[:2]
        roi_size = min(h, w) // 2
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2

        # Draw ROI rectangle
        cv2.rectangle(display_frame, (roi_x, roi_y),
                      (roi_x + roi_size, roi_y + roi_size),
                      (0, 255, 0), 2)

        # Crop ROI for prediction
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        if roi.size == 0:
            continue

        # Preprocess the ROI and make prediction
        input_tensor = preprocess_frame(roi)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get top 3 predictions
        conf_values, indices = torch.topk(probabilities, 3)
        current_predictions = [(class_names[idx.item()], conf.item()) for idx, conf in zip(indices[0], conf_values[0])]

        # Temporal smoothing - keep a history of predictions
        pred_history.append(current_predictions)
        if len(pred_history) > history_len:
            pred_history.pop(0)

        # Calculate the most common prediction in history
        if len(pred_history) >= 3:  # Need at least 3 frames for smoothing
            # Get the most frequent top prediction
            top_classes = [p[0][0] for p in pred_history]  # Extract top class name from each frame
            counter = Counter(top_classes)
            most_common = counter.most_common(1)[0][0]  # Get most common class

            # Find average confidence for this class across frames
            avg_conf = 0
            count = 0
            for preds in pred_history:
                for class_name, conf in preds:
                    if class_name == most_common:
                        avg_conf += conf
                        count += 1
                        break

            if count > 0:
                avg_conf /= count
                top_predictions = [(most_common, avg_conf)]

                # Add next two most common classes
                for class_name, count in counter.most_common(3)[1:]:
                    if len(top_predictions) >= 3:
                        break
                    # Find average confidence for this class
                    avg_conf = 0
                    count = 0
                    for preds in pred_history:
                        for c_name, conf in preds:
                            if c_name == class_name:
                                avg_conf += conf
                                count += 1
                                break

                    if count > 0:
                        avg_conf /= count
                        top_predictions.append((class_name, avg_conf))
            else:
                top_predictions = current_predictions
        else:
            top_predictions = current_predictions

        # Get prediction color based on confidence (green high, yellow medium, red low)
        top_conf = top_predictions[0][1]
        if top_conf > 0.7:
            color = (0, 255, 0)  # Green
        elif top_conf > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Draw a black background for the prediction text
        cv2.rectangle(display_frame, (10, 10), (300, 120), (0, 0, 0), -1)

        # Add prediction text - only if confidence is above threshold
        if top_conf > confidence_threshold:
            cv2.putText(display_frame, f"Object: {top_predictions[0][0]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(display_frame, f"Confidence: {top_predictions[0][1]:.2f}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        else:
            cv2.putText(display_frame, "No confident prediction",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Add other top predictions
        y_pos = 100
        for i in range(1, len(top_predictions)):
            if top_predictions[i][1] > confidence_threshold:
                pred_text = f"{i + 1}. {top_predictions[i][0]} ({top_predictions[i][1]:.2f})"
                cv2.putText(display_frame, pred_text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 30

        # Add FPS counter
        if show_fps:
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                        (display_frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add help text
        if show_help:
            help_text = [
                "Press 'q' to quit",
                "Press 's' to save snapshot",
                "Press 'h' to toggle help",
                "Press 'f' to toggle FPS display"
            ]
            for i, text in enumerate(help_text):
                cv2.putText(display_frame, text,
                            (10, display_frame.shape[0] - 10 - i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Object Recognition", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame with prediction
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            snapshot_path = os.path.join(snapshots_dir,
                                         f"{timestamp}_{top_predictions[0][0]}_{top_conf:.2f}.jpg")
            cv2.imwrite(snapshot_path, frame)
            print(f"Snapshot saved to {snapshot_path}")
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('f'):
            show_fps = not show_fps

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam recognition stopped.")


def main():
    """
    Load the trained weights and launch the real-time webcam demo.
    """
    if not os.path.exists(MODEL_FILE):
        print(f"Error: weights file '{MODEL_FILE}' not found.")
        print("Download it from the project Releases, or run 'python train.py' first.")
        return

    model = ImprovedCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print(f"Loaded model from {MODEL_FILE}")

    run_webcam_recognition(model, CLASS_NAMES)


if __name__ == "__main__":
    main()
