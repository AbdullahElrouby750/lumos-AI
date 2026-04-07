import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from deepface import DeepFace
import numpy as np
import os

class FaceRecognizer:
    """
    Refactored face recognizer as a callable class.
    Loads known faces, initializes detector and model, provides recognize method.
    """

    def __init__(self, brain_file="nova_brain.pkl", model_path='blaze_face_short_range.tflite'):
        """
        Initialize the FaceRecognizer.
        """
        self.known_faces = {}
        self.detector = None
        self.facenet_model = None
        self.KNOWN_WIDTH = 14.0  # cm
        self.FOCAL_LENGTH = 678.57  # Preserved calibrated value

        # Load brain
        try:
            if os.path.exists(brain_file):
                with open(brain_file, "rb") as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces from the brain.")
            else:
                print("Brain file not found. No known faces loaded.")
        except Exception as e:
            print(f"Error loading brain: {e}")

        # Initialize detector
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
        except Exception as e:
            print(f"Error initializing detector: {e}")

        # Load Facenet model
        try:
            self.facenet_model = DeepFace.build_model("Facenet512")
        except Exception as e:
            print(f"Error loading Facenet model: {e}")

    def recognize_face_crop(self, face_crop):
        """
        Takes a raw crop, ALIGNS IT mathematically, extracts the embedding,
        and compares it against the database. Heavy, but highly accurate.
        """
        if not self.known_faces:
            return "Unknown"

        try:
            # THE RIGHT WAY: Let DeepFace handle the alignment and extraction on the live crop
            # enforce_detection=False because we already know it's a face crop
            results = DeepFace.represent(
                img_path=face_crop, 
                model_name="Facenet512", 
                enforce_detection=False, 
                align=True
            )
            
            current_embedding = results[0]["embedding"]

            # Compare against the database
            best_match = "Unknown"
            min_dist = 0.4  # Stricter threshold because our data is cleaner now!

            for name, embeddings in self.known_faces.items():
                for db_emb in embeddings:
                    # Cosine distance math
                    a = np.matmul(current_embedding, db_emb)
                    b = np.sum(np.multiply(current_embedding, current_embedding))
                    c = np.sum(np.multiply(db_emb, db_emb))
                    dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

                    if dist < min_dist:
                        min_dist = dist
                        best_match = name

            return best_match

        except Exception as e:
            print(f"Alignment/Recognition error in engine: {e}")
            return "Unknown"

    def close(self):
        """
        Close the detector.
        """
        if self.detector:
            self.detector.close()