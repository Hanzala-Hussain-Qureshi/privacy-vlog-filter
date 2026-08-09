"""
STONE 2 (MediaPipe Tasks API version): Live blur for unknown faces
------------------------------------------------------------------------
Detects every face in the webcam feed using the new FaceLandmarker task,
builds the same normalized fingerprint used in enrollment, and compares
it to known faces using Euclidean distance.

- Known face  -> green box, not blurred
- Unknown face -> heavily blurred

Requires: face_landmarker.task in the same folder as this script.
Run enroll_faces_tasks.py FIRST so known_faces.pkl exists.
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import time

KNOWN_FACES_FILE = "known_faces.pkl"
MODEL_PATH = "face_landmarker.task"
MATCH_THRESHOLD = 1.2  # tuned based on measured self-distance (~0.75-1.5)

def landmarks_to_embedding(landmarks, image_width, image_height):
    points = np.array([[lm.x * image_width, lm.y * image_height] for lm in landmarks])
    center = points.mean(axis=0)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    points = points / scale
    return points.flatten()

def get_bounding_box(landmarks, image_width, image_height, padding=15):
    xs = [lm.x * image_width for lm in landmarks]
    ys = [lm.y * image_height for lm in landmarks]
    left, right = int(min(xs)) - padding, int(max(xs)) + padding
    top, bottom = int(min(ys)) - padding, int(max(ys)) + padding
    left, top = max(0, left), max(0, top)
    right, bottom = min(image_width, right), min(image_height, bottom)
    return left, top, right, bottom

def load_known_faces():
    if not os.path.exists(KNOWN_FACES_FILE):
        raise FileNotFoundError("No known_faces.pkl found. Run enroll_faces_tasks.py first.")
    with open(KNOWN_FACES_FILE, "rb") as f:
        return pickle.load(f)

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Can't find {MODEL_PATH} — make sure you downloaded it into this folder."
        )

    known_data = load_known_faces()
    known_embeddings = known_data["embeddings"]
    known_names = known_data["names"]
    print(f"Loaded known faces: {known_names}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5   # allow multiple people in frame
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    video = cv2.VideoCapture(0)
    last_print_time = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        for landmarks in result.face_landmarks:
            embedding = landmarks_to_embedding(landmarks, w, h)
            left, top, right, bottom = get_bounding_box(landmarks, w, h)

            is_known = False
            name = "Unknown"
            if len(known_embeddings) > 0:
                distances = [np.linalg.norm(embedding - k) for k in known_embeddings]
                best_index = int(np.argmin(distances))
                if time.time() - last_print_time > 1.0:
                    print(f"Closest match: {known_names[best_index]}  distance = {distances[best_index]:.4f}")
                    last_print_time = time.time()
                if distances[best_index] < MATCH_THRESHOLD:
                    is_known = True
                    name = known_names[best_index]

            if is_known:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                face_region = frame[top:bottom, left:right]
                if face_region.size > 0:
                    blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                    frame[top:bottom, left:right] = blurred

        cv2.imshow("Privacy Filter - 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()