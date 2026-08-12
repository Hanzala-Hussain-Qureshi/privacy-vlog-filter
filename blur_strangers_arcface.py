"""
STONE 2 (ArcFace version + temporal smoothing): Live blur for unknown faces
----------------------------------------------------------------------------
This is the fix for the "Snapchat twitch" problem from the very start of
this project: every frame used to be treated as a brand new, independent
detection with zero memory of previous frames. If detection confidence
dipped for even one frame, the box vanished and reappeared - the exact
flicker you noticed.

This version adds simple TRACKING between frames:
  - Each detected face becomes a "track" that persists across frames
  - A track's box position is SMOOTHED (blended with its previous
    position) instead of jumping instantly - reduces jitter
  - If a track isn't re-detected for a few frames (a brief dropout),
    it's kept alive using its last known position for a short "grace
    period" instead of instantly disappearing - reduces flicker
  - Only if a track is missing for longer than the grace period does it
    actually get removed

This is a lightweight, from-scratch version of the same idea we
discussed at the very beginning of this project: using prediction/motion
smoothing (like a Kalman filter) to bridge gaps in tracking, rather than
throwing away all information the instant one frame is uncertain.

Also increases webcam capture resolution, since faces further from the
camera are just smaller patches of pixels - more resolution gives the
detector more detail to work with at distance.

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

MATCH_THRESHOLD = 0.45

# --- Tracking / smoothing settings ---
SMOOTHING_ALPHA = 0.4      # 0 = no smoothing (jumpy), 1 = frozen (no update). 0.4 is a good middle ground.
GRACE_PERIOD_FRAMES = 12   # how many frames a face can go undetected before its track is dropped (~0.4s at 30fps)
MAX_MATCH_DISTANCE = 120   # pixels; how close a new detection's center must be to an existing track to count as "the same face"

# --- Camera capture settings ---
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720


class FaceTrack:
    """Represents one face being followed across multiple frames."""
    _next_id = 0

    def __init__(self, box, is_known, name):
        self.id = FaceTrack._next_id
        FaceTrack._next_id += 1
        self.box = box              # (left, top, right, bottom) - smoothed
        self.is_known = is_known
        self.name = name
        self.frames_since_seen = 0

    def center(self):
        left, top, right, bottom = self.box
        return ((left + right) / 2, (top + bottom) / 2)

    def update(self, new_box, is_known, name):
        # Blend the new box position with the old one instead of snapping directly to it
        l0, t0, r0, b0 = self.box
        l1, t1, r1, b1 = new_box
        a = SMOOTHING_ALPHA
        self.box = (
            int(l0 * (1 - a) + l1 * a),
            int(t0 * (1 - a) + t1 * a),
            int(r0 * (1 - a) + r1 * a),
            int(b0 * (1 - a) + b1 * a),
        )
        self.is_known = is_known
        self.name = name
        self.frames_since_seen = 0


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
    return float(np.dot(a, b))


def load_known_faces():
    if not os.path.exists(KNOWN_FACES_FILE):
        raise FileNotFoundError("No known_faces.pkl found. Run enroll_faces_arcface.py first.")
    with open(KNOWN_FACES_FILE, "rb") as f:
        return pickle.load(f)


def match_detection_to_track(detection_box, tracks):
    """Find the existing track whose center is closest to this new detection, if close enough."""
    d_left, d_top, d_right, d_bottom = detection_box
    d_center = ((d_left + d_right) / 2, (d_top + d_bottom) / 2)

    best_track = None
    best_dist = MAX_MATCH_DISTANCE
    for track in tracks:
        t_center = track.center()
        dist = ((d_center[0] - t_center[0]) ** 2 + (d_center[1] - t_center[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_track = track
    return best_track


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
        num_faces=5,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    arc_session = ort.InferenceSession(ARCFACE_MODEL)

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    tracks = []
    last_print_time = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        matched_tracks = set()

        # --- Process this frame's raw detections ---
        for landmarks in result.face_landmarks:
            box = get_bounding_box(landmarks, w, h)
            left, top, right, bottom = box
            face_crop = frame[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            embedding = get_arcface_embedding(arc_session, face_crop)

            is_known = False
            name = "Unknown"
            if len(known_embeddings) > 0:
                similarities = [cosine_similarity(embedding, k) for k in known_embeddings]
                best_index = int(np.argmax(similarities))

                if time.time() - last_print_time > 1.0:
                    print(f"Closest match: {known_names[best_index]}  similarity = {similarities[best_index]:.4f}")
                    last_print_time = time.time()

                if similarities[best_index] > MATCH_THRESHOLD:
                    is_known = True
                    name = known_names[best_index]

            # --- Match this detection to an existing track, or start a new one ---
            track = match_detection_to_track(box, tracks)
            if track is not None:
                track.update(box, is_known, name)
                matched_tracks.add(track.id)
            else:
                new_track = FaceTrack(box, is_known, name)
                tracks.append(new_track)
                matched_tracks.add(new_track.id)

        # --- Age out tracks that weren't matched this frame ---
        still_alive = []
        for track in tracks:
            if track.id not in matched_tracks:
                track.frames_since_seen += 1
            if track.frames_since_seen <= GRACE_PERIOD_FRAMES:
                still_alive.append(track)
        tracks = still_alive

        # --- Draw every currently alive track (even ones not re-detected THIS exact frame) ---
        for track in tracks:
            left, top, right, bottom = track.box
            left, top = max(0, left), max(0, top)
            right, bottom = min(w, right), min(h, bottom)
            if right <= left or bottom <= top:
                continue

            if track.is_known:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, track.name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                face_region = frame[top:bottom, left:right]
                if face_region.size > 0:
                    blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                    frame[top:bottom, left:right] = blurred

        cv2.imshow("Privacy Filter (ArcFace + tracking) - 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()