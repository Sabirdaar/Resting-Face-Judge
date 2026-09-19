import sys
from pathlib import Path
import cv2
import joblib
import numpy as np

# Ensure project root is in sys.path and register classes for joblib unpickling
base_dir = Path(__file__).resolve().parent
root_dir = base_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.train_random_forest import (
        RandomForestClassifier,
        PureDecisionTree,
        DecisionNode,
    )

    sys.modules["__main__"].RandomForestClassifier = RandomForestClassifier
    sys.modules["__main__"].PureDecisionTree = PureDecisionTree
    sys.modules["__main__"].DecisionNode = DecisionNode
except ImportError:
    pass

EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def preprocess_face_crop(face_crop: np.ndarray, target_size=(48, 48)) -> np.ndarray:
    """Converts cropped BGR face image to 48x48 grayscale model input."""
    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(gray_crop, target_size, interpolation=cv2.INTER_AREA)
    return face_resized


def main():
    # Load Caffe Face Detector
    prototxt_path = base_dir / "vision" / "models" / "deploy.prototxt"
    caffemodel_path = (
        base_dir / "vision" / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
    )

    if not prototxt_path.exists():
        prototxt_path = root_dir / "src" / "vision" / "models" / "deploy.prototxt"
    if not caffemodel_path.exists():
        caffemodel_path = (
            root_dir
            / "src"
            / "vision"
            / "models"
            / "res10_300x300_ssd_iter_140000.caffemodel"
        )

    net = cv2.dnn.readNetFromCaffe(
        str(prototxt_path),
        str(caffemodel_path),
    )

    # Load Random Forest Classifier
    rf_model_path = root_dir / "models" / "random_forest_fer2013.joblib"
    rf_model = None
    if rf_model_path.exists():
        try:
            rf_model = joblib.load(rf_model_path)
            print(f"Loaded Random Forest model from {rf_model_path}")
        except Exception as e:
            print(f"Error loading model from {rf_model_path}: {e}")
    else:
        print(
            f"Warning: Model file not found at {rf_model_path}. Running detection only."
        )

    CONFIDENCE_THRESHOLD = 0.5  # Face detection confidence threshold

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)

                # Ensure bounding box coordinates stay within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    # 1. Detect & crop face -> 48x48 grayscale for model input
                    face_48x48 = preprocess_face_crop(face_roi, target_size=(48, 48))

                    # 2. Predict emotion using Random Forest Classifier
                    predicted_emotion = "Face Detected"
                    if rf_model is not None:
                        face_flat = face_48x48.reshape(1, -1)
                        pred_idx = rf_model.predict(face_flat)[0]
                        if hasattr(rf_model, "classes"):
                            predicted_emotion = str(rf_model.classes[pred_idx])
                        else:
                            predicted_emotion = EMOTION_CLASSES[pred_idx]

                    # 3. Draw bounding box and emotion label on frame
                    label_text = f"{predicted_emotion} ({confidence:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # 4. Overlay 48x48 input crop preview on top-left of video feed
                    preview_size = 96
                    crop_display = cv2.cvtColor(face_48x48, cv2.COLOR_GRAY2BGR)
                    crop_preview = cv2.resize(
                        crop_display,
                        (preview_size, preview_size),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    frame[10 : 10 + preview_size, 10 : 10 + preview_size] = crop_preview
                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (10 + preview_size, 10 + preview_size),
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"Input: {predicted_emotion}",
                        (10, 25 + preview_size),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )

        cv2.imshow("Resting Face Judge — End-to-End Live Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
