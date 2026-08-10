"""
STONE 2 (ArcFace version): Live blur for unknown faces
------------------------------------------------------------
Uses MediaPipe to find and crop each face, ArcFace to turn each crop
into a real 512-number pose-robust identity embedding, then compares
using COSINE SIMILARITY (not raw distance) - remember our earlier
discussion about why angle is more stable than gap under lighting/pose
changes? Same idea applies here.

Cosine similarity ranges roughly -1 to 1:
  - Close to 1  -> very likely the SAME person
  - Close to 0 or negative -> likely a DIFFERENT person

Requires in the same folder:
  - face_landmarker.task
  - arcfaceresnet100-8.onnx
  - known_faces.pkl (created by enroll_faces_arcface.py)

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import onnxruntime as ort
import pickle
import os
import time

KNOWN_FACES_FILE = "known_faces.pkl"
LANDMARKER_MODEL = "face_landmarker.task"
ARCFACE_MODEL = "arcfaceresnet100-8.onnx"

# Cosine similarity threshold. Higher = stricter matching.
# Typical starting point for ArcFace is around 0.4-0.5 -- we'll tune with real numbers.
MATCH_THRESHOLD = 0.45

def get_bounding_box(landmarks, image_width, image_height, padding=20):
    xs = [lm.x * image_width for lm in landmarks]
    ys = [lm.y * image_height for lm in landmarks]
    left, right = int(min(xs)) - padding, int(max(xs)) + padding
    top, bottom = int(min(ys)) - padding, int(max(ys)) + padding
    left, top = max(0, left), max(0, top)
    right, bottom = min(image_width, right), min(image_height, bottom)
    return left, top, right, bottom

def get_arcface_embedding(session, face_crop_bgr):
    face_resized = cv2.resize(face_crop_bgr, (112, 112))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_norm = (face_rgb.astype(np.float32) - 127.5) / 128.0
    face_chw = np.transpose(face_norm, (2, 0, 1))
    face_input = np.expand_dims(face_chw, axis=0)

    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face_input})[0]
    embedding = embedding.flatten()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def cosine_similarity(a, b):
    # Both vectors are already normalized (length 1), so this is just a dot product
    return float(np.dot(a, b))

def load_known_faces():
    if not os.path.exists(KNOWN_FACES_FILE):
        raise FileNotFoundError("No known_faces.pkl found. Run enroll_faces_arcface.py first.")
    with open(KNOWN_FACES_FILE, "rb") as f:
        return pickle.load(f)

def main():
    for path in (LANDMARKER_MODEL, ARCFACE_MODEL):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can't find {path} — make sure it's in this folder.")

    known_data = load_known_faces()
    known_embeddings = known_data["embeddings"]
    known_names = known_data["names"]
    print(f"Loaded known faces: {known_names}")

    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    arc_session = ort.InferenceSession(ARCFACE_MODEL)

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
            left, top, right, bottom = get_bounding_box(landmarks, w, h)
            face_crop = frame[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            embedding = get_arcface_embedding(arc_session, face_crop)

            is_known = False
            name = "Unknown"
            if len(known_embeddings) > 0:
                similarities = [cosine_similarity(embedding, k) for k in known_embeddings]
                best_index = int(np.argmax(similarities))  # HIGHEST similarity wins

                if time.time() - last_print_time > 1.0:
                    print(f"Closest match: {known_names[best_index]}  similarity = {similarities[best_index]:.4f}")
                    last_print_time = time.time()

                if similarities[best_index] > MATCH_THRESHOLD:
                    is_known = True
                    name = known_names[best_index]

            if is_known:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                blurred = cv2.GaussianBlur(face_crop, (99, 99), 30)
                frame[top:bottom, left:right] = blurred

        cv2.imshow("Privacy Filter (ArcFace) - 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()