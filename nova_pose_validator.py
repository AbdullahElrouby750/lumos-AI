import os
import collections
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Tuning constants ──────────────────────────────────────────────────────────
# Number of frames whose ratios are averaged before a pose decision is made.
# Larger = smoother but slower to respond.  5 frames at 30 FPS = ~167 ms lag,
# which is imperceptible to the user but eliminates single-frame landmark jitter.
_SMOOTHING_WINDOW = 5

# How many of the last _SMOOTHING_WINDOW averaged frames must agree on the
# same pose label before we accept it as valid.
# e.g. 3/5 means the averaged pose must be stable for 3 consecutive frames.
_CONSENSUS_REQUIRED = 3

# Ratio thresholds – kept identical to the original values so behaviour is
# unchanged when the signal is clean.
_YAW_HIGH  = 1.3   # yaw_ratio  > this  → head turned right
_YAW_LOW   = 0.6   # yaw_ratio  < this  → head turned left
_PITCH_HIGH = 1.3  # pitch_ratio > this → head tilted down
_PITCH_LOW  = 0.6  # pitch_ratio < this → head tilted up
# ─────────────────────────────────────────────────────────────────────────────


class PoseValidator:
    """
    Validates face pose using MediaPipe FaceLandmarker with temporal smoothing.

    Instead of classifying a single-frame ratio (fragile under jitter / low
    light), it maintains a rolling deque of (yaw_ratio, pitch_ratio) pairs.
    The decision is made on the *moving average* of the last _SMOOTHING_WINDOW
    valid frames, and the pose must remain consistent for _CONSENSUS_REQUIRED
    consecutive averaged-frames before returning True.

    The validator is stateful and tied to one enrollment session step.
    Call reset() at the start of each new step so the history from the
    previous pose does not bleed into the next one.
    """

    def __init__(self, model_path: str = 'face_landmarker.task'):
        self.model_path = model_path
        self._ensure_model_exists()

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[PoseValidator] Error initialising FaceLandmarker: {e}")
            self.landmarker = None

        # ── BUG 2 FIX: rolling buffers ────────────────────────────────────────
        # Stores raw (yaw_ratio, pitch_ratio) floats from each valid frame.
        self._ratio_buffer: collections.deque = collections.deque(
            maxlen=_SMOOTHING_WINDOW
        )
        # Stores the smoothed pose label computed each frame.
        self._pose_label_buffer: collections.deque = collections.deque(
            maxlen=_SMOOTHING_WINDOW
        )
        # ──────────────────────────────────────────────────────────────────────

    # ── Public API ────────────────────────────────────────────────────────────
    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def reset(self):
        """
        Clear history buffers.  Must be called at the start of each new
        enrollment step so stale ratios from the previous pose don't influence
        the new one.
        """
        self._ratio_buffer.clear()
        self._pose_label_buffer.clear()

    def validate_pose(self, frame, required_pose: str):
        """
        Returns (True, "Success") only when the smoothed pose has matched
        required_pose for _CONSENSUS_REQUIRED consecutive averaged-frames.

        Returns (False, feedback_str) in all other cases.
        """
        if self.landmarker is None:
            return False, "Validator not initialised."

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            # No face visible: clear buffers so we don't carry stale data
            self._ratio_buffer.clear()
            self._pose_label_buffer.clear()
            return False, "Please ensure your face is fully visible."

        landmarks = results.face_landmarks[0]

        # ── Extract T-Zone geometry (same landmarks as original) ──────────────
        brow_y  = landmarks[168].y
        nose_y  = landmarks[1].y
        chin_y  = landmarks[152].y
        nose_x  = landmarks[1].x

        face_height = abs(chin_y - brow_y)

        # Occlusion check: collapsed lip gap means a hand/object is in the way
        upper_lip_y = landmarks[13].y
        lower_lip_y = landmarks[14].y
        lip_gap     = abs(upper_lip_y - lower_lip_y)
        if lip_gap < (face_height * 0.002):
            self._ratio_buffer.clear()
            self._pose_label_buffer.clear()
            return False, "Please uncover your mouth."

        left_edge_x  = landmarks[234].x
        right_edge_x = landmarks[454].x

        upper_face = abs(nose_y - brow_y)
        lower_face = abs(chin_y - nose_y)
        left_side  = abs(nose_x - left_edge_x)
        right_side = abs(right_edge_x - nose_x)

        # Raw single-frame ratios
        pitch_ratio = upper_face / (lower_face + 1e-6)
        yaw_ratio   = left_side  / (right_side + 1e-6)

        # ── BUG 2 FIX: push into rolling buffer ───────────────────────────────
        self._ratio_buffer.append((yaw_ratio, pitch_ratio))

        # Compute moving average over whatever frames we have so far
        n = len(self._ratio_buffer)
        avg_yaw   = sum(r[0] for r in self._ratio_buffer) / n
        avg_pitch = sum(r[1] for r in self._ratio_buffer) / n

        
        print(f"Ratios: yaw={avg_yaw:.2f}, pitch={avg_pitch:.2f} (buffer size={n})")
        print(f"Recent labels: {list(self._pose_label_buffer)}")    
        print(f"Required pose: {required_pose}")
        print("-" * 30)
        print(f"Debug info: {self._ratio_buffer}")
        print("-" * 30)
        
        # Classify the *averaged* signal
        if avg_yaw > _YAW_HIGH:
            smoothed_pose = "right"
        elif avg_yaw < _YAW_LOW:
            smoothed_pose = "left"
        elif avg_pitch > _PITCH_HIGH:
            smoothed_pose = "down"
        elif avg_pitch < _PITCH_LOW:
            smoothed_pose = "up"
        else:
            smoothed_pose = "straight"

        self._pose_label_buffer.append(smoothed_pose)
        # ──────────────────────────────────────────────────────────────────────

        # ── Consensus gate ────────────────────────────────────────────────────
        # Count how many of the most-recent label frames match the required pose
        recent_labels = list(self._pose_label_buffer)
        matching      = sum(1 for lbl in recent_labels if lbl == required_pose)

        if matching >= _CONSENSUS_REQUIRED:
            # Success: clear history so the next step starts fresh
            self.reset()
            return True, "Success"

        # Not there yet: give the user a helpful nudge using the smoothed pose
        if smoothed_pose != required_pose:
            return False, f"Please look {required_pose}."
        else:
            # Smoothed pose is correct but we haven't hit consensus yet —
            # the user is in position but we're still accumulating frames.
            # Give neutral feedback so the TTS doesn't spam "please look right"
            # when they're already looking right.
            return False, ""   # empty string → caller should not re-speak

    def close(self):
        if self.landmarker:
            self.landmarker.close()