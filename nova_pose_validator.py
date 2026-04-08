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
            # Initialize the modern Tasks API (Same standard as nova_face_detector)
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
        """Automatically downloads the FaceLandmarker model if it's missing."""
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def validate_pose(self, frame, required_pose):
        """
        Validates the pose in the frame against the required pose.
        Returns (is_valid: bool, feedback_message: str)
        """
        if self.landmarker is None:
            return False, "Validator not initialized."

        # Convert frame to mp.Image (Tasks API standard)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect(mp_image)

        # If it can't find a face mesh at all, it's occluded or off-camera
        if not results.face_landmarks:
            return False, "Please ensure your eyes, nose, and mouth are visible."

        landmarks = results.face_landmarks[0]

        # Extract T-Zone landmarks (x, y coordinates)
        nose = np.array([landmarks[1].x, landmarks[1].y])
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        upper_lip = np.array([landmarks[13].x, landmarks[13].y])
        lower_lip = np.array([landmarks[14].x, landmarks[14].y])

        # Average eye and mouth positions
        left_eye_center = (left_eye_inner + left_eye_outer) / 2
        right_eye_center = (right_eye_inner + right_eye_outer) / 2
        mouth_center = (upper_lip + lower_lip) / 2

        # Yaw: Horizontal asymmetry (left/right)
        dist_nose_left = np.linalg.norm(nose - left_eye_center)
        dist_nose_right = np.linalg.norm(nose - right_eye_center)
        yaw_ratio = dist_nose_left / (dist_nose_right + 1e-6)  # Avoid division by zero

        # Pitch: Vertical positioning (up/down)
        eye_nose_dist = np.abs(nose[1] - (left_eye_center[1] + right_eye_center[1]) / 2)
        nose_mouth_dist = np.abs(mouth_center[1] - nose[1])
        pitch_ratio = eye_nose_dist / (nose_mouth_dist + 1e-6)

        # Determine calculated pose (Mirror-Corrected Math)
        if yaw_ratio > 1.2:  
            calculated_pose = "right"
        elif yaw_ratio < 0.8:  
            calculated_pose = "left"
        elif pitch_ratio > 1.2:  
            calculated_pose = "down"
        elif pitch_ratio < 0.8:  
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