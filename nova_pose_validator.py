import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseValidator:
    """Validates face pose using the modern MediaPipe FaceLandmarker for T-Zone landmarks."""

    def __init__(self, model_path='face_landmarker.task'):
        self.model_path = model_path
        self._ensure_model_exists()
        
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing FaceLandmarker: {e}")
            self.landmarker = None

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def validate_pose(self, frame, required_pose):
        if self.landmarker is None:
            return False, "Validator not initialized."

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks:
            return False, "Please ensure your face is fully visible."

        landmarks = results.face_landmarks[0]

        # --- OCCLUSION CHECK (The Hand-over-mouth fix) ---
        # If a hand covers the mouth, the AI hallucinates the lips, often crushing them together.
        upper_lip = landmarks[13].y
        lower_lip = landmarks[14].y
        if abs(upper_lip - lower_lip) < 0.001: 
            return False, "Please uncover your mouth."

        # --- ANATOMICAL POSE MATH ---
        # Pitch (Up/Down) -> Brow (168), Nose Tip (1), Chin (152)
        brow_y = landmarks[168].y
        nose_y = landmarks[1].y
        chin_y = landmarks[152].y
        
        # Yaw (Left/Right) -> Left Cheek Edge (234), Nose Tip (1), Right Cheek Edge (454)
        left_edge_x = landmarks[234].x
        nose_x = landmarks[1].x
        right_edge_x = landmarks[454].x

        # Calculate distances
        upper_face = abs(nose_y - brow_y)
        lower_face = abs(chin_y - nose_y)
        left_side = abs(nose_x - left_edge_x)
        right_side = abs(right_edge_x - nose_x)

        # Ratios
        pitch_ratio = upper_face / (lower_face + 1e-6)
        yaw_ratio = left_side / (right_side + 1e-6)

        # Determine calculated pose (Mirror-Corrected Math)
        # Using safe thresholds (1.35 and 0.7) to allow for natural resting faces
        if yaw_ratio > 1.35:  
            calculated_pose = "right"
        elif yaw_ratio < 0.7:  
            calculated_pose = "left"
        elif pitch_ratio > 1.35:  
            calculated_pose = "down"
        elif pitch_ratio < 0.7:  
            calculated_pose = "up"
        else:
            calculated_pose = "straight"

        # Compare to required pose
        if calculated_pose == required_pose:
            return True, "Success"
        else:
            return False, f"Please look {required_pose}."

    def close(self):
        if self.landmarker:
            self.landmarker.close()