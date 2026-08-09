"""
STONE 1 (MediaPipe Tasks API version): Enroll known faces
--------------------------------------------------------------
Uses MediaPipe's new FaceLandmarker task to find 478 landmark points on
a face, then builds our own simple "fingerprint" (embedding) by
normalizing those points and flattening them into one long list of
numbers.

Requires: face_landmarker.task in the same folder as this script.
Press 's' to save the current face under a name. Press 'q' to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os

KNOWN_FACES_FILE = "known_faces.pkl"
MODEL_PATH = "face_landmarker.task"

def landmarks_to_embedding(landmarks, image_width, image_height):
    points = np.array([[lm.x * image_width, lm.y * image_height] for lm in landmarks])
    center = points.mean(axis=0)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    points = points / scale
    return points.flatten()

def load_known_faces():
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {"embeddings": [], "names": []}

def save_known_faces(data):
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(data, f)

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Can't find {MODEL_PATH} — make sure you downloaded it into this folder."
        )

    known_data = load_known_faces()
    print(f"Currently enrolled: {known_data['names']}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    video = cv2.VideoCapture(0)
    print("\nPress 's' to save the current face. Press 'q' to quit.\n")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            left, top = int(min(xs)), int(min(ys))
            right, bottom = int(max(xs)), int(max(ys))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Enroll Faces - 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if not result.face_landmarks:
                print("No face detected — try again.")
                continue
            if len(result.face_landmarks) > 1:
                print("More than one face detected — enroll one person at a time.")
                continue

            embedding = landmarks_to_embedding(result.face_landmarks[0], w, h)
            name = input("Enter a name for this face: ").strip()
            known_data["embeddings"].append(embedding)
            known_data["names"].append(name)
            save_known_faces(known_data)
            print(f"Saved face for '{name}'. Total enrolled: {known_data['names']}")

        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()