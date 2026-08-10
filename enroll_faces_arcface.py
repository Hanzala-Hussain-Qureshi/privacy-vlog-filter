"""
STONE 1 (ArcFace version): Enroll known faces
--------------------------------------------------
Uses MediaPipe (face_landmarker.task) to FIND the face and crop it out,
then feeds that crop into ArcFace (arcfaceresnet100-8.onnx) - a real
pretrained face recognition model - to get a proper, pose-robust
512-number fingerprint (embedding).

Requires in the same folder:
  - face_landmarker.task
  - arcfaceresnet100-8.onnx

Press 's' to save the current face under a name. Press 'q' to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import onnxruntime as ort
import pickle
import os

KNOWN_FACES_FILE = "known_faces.pkl"
LANDMARKER_MODEL = "face_landmarker.task"
ARCFACE_MODEL = "arcfaceresnet100-8.onnx"

def get_bounding_box(landmarks, image_width, image_height, padding=20):
    xs = [lm.x * image_width for lm in landmarks]
    ys = [lm.y * image_height for lm in landmarks]
    left, right = int(min(xs)) - padding, int(max(xs)) + padding
    top, bottom = int(min(ys)) - padding, int(max(ys)) + padding
    left, top = max(0, left), max(0, top)
    right, bottom = min(image_width, right), min(image_height, bottom)
    return left, top, right, bottom

def get_arcface_embedding(session, face_crop_bgr):
    """
    Preprocess a cropped face image and run it through ArcFace to get
    a 512-number identity embedding.
    """
    face_resized = cv2.resize(face_crop_bgr, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # ArcFace expects pixel values roughly normalized to [-1, 1]
    face_norm = (face_rgb.astype(np.float32) - 127.5) / 128.0

    # Reorder to (channels, height, width) and add a batch dimension
    face_chw = np.transpose(face_norm, (2, 0, 1))
    face_input = np.expand_dims(face_chw, axis=0)

    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face_input})[0]
    embedding = embedding.flatten()

    # Normalize the embedding vector itself -> makes cosine similarity comparison clean
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def load_known_faces():
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {"embeddings": [], "names": []}

def save_known_faces(data):
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump(data, f)

def main():
    for path in (LANDMARKER_MODEL, ARCFACE_MODEL):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can't find {path} — make sure it's in this folder.")

    known_data = load_known_faces()
    print(f"Currently enrolled: {known_data['names']}")

    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    arc_session = ort.InferenceSession(ARCFACE_MODEL)

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

        box = None
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            box = get_bounding_box(landmarks, w, h)
            left, top, right, bottom = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Enroll Faces (ArcFace) - 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if not result.face_landmarks:
                print("No face detected — try again.")
                continue
            if len(result.face_landmarks) > 1:
                print("More than one face detected — enroll one person at a time.")
                continue

            left, top, right, bottom = box
            face_crop = frame[top:bottom, left:right]
            if face_crop.size == 0:
                print("Face crop was empty — try repositioning.")
                continue

            embedding = get_arcface_embedding(arc_session, face_crop)

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