import cv2
import mediapipe as mp
import numpy as np

class PoseValidator:
    """Validates face pose using MediaPipe FaceMesh for T-Zone landmarks."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def validate_pose(self, frame, required_pose):
        """
        Validates the pose in the frame against the required pose.
        Returns (is_valid: bool, feedback_message: str)
        """
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return False, "Please ensure your eyes, nose, and mouth are visible."

        landmarks = results.multi_face_landmarks[0].landmark

        # T-Zone landmarks indices (approximate for key points)
        # Nose tip: 1
        # Left eye inner: 133, outer: 33
        # Right eye inner: 362, outer: 263
        # Upper lip: 13, lower lip: 14

        t_zone_indices = [1, 133, 33, 362, 263, 13, 14]

        # Check if all T-Zone landmarks are visible (not occluded)
        for idx in t_zone_indices:
            if landmarks[idx].visibility < 0.5:  # Threshold for visibility
                return False, "Please ensure your eyes, nose, and mouth are visible."

        # Calculate pose
        nose = np.array([landmarks[1].x, landmarks[1].y])
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        upper_lip = np.array([landmarks[13].x, landmarks[13].y])
        lower_lip = np.array([landmarks[14].x, landmarks[14].y])

        # Average eye positions
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
            # Nose is closer to the right side of the image -> Person is looking Right
            calculated_pose = "right"
        elif yaw_ratio < 0.8:  
            # Nose is closer to the left side of the image -> Person is looking Left
            calculated_pose = "left"
        elif pitch_ratio > 1.2:  
            # Nose is closer to the mouth -> Person is looking Down
            calculated_pose = "down"
        elif pitch_ratio < 0.8:  
            # Nose is closer to the eyes -> Person is looking Up
            calculated_pose = "up"
        else:
            calculated_pose = "straight"

        # Compare to required pose
        if calculated_pose == required_pose:
            return True, "Success"
        else:
            return False, f"Please look {required_pose}."