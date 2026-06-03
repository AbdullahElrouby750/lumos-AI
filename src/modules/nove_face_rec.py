import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from deepface import DeepFace
import numpy as np
import os
import threading

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
        self._brain_lock = threading.Lock()
        self.detector = None
        self.facenet_model = None
        self.KNOWN_WIDTH = 14.0  # cm
        self.FOCAL_LENGTH = 678.57  # Preserved calibrated value
        self.brain_file = brain_file
        
        # 1. Load brain using the new hot-reload method
        self.load_brain()

        # 2. Initialize detector
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
        except Exception as e:
            print(f"Error initializing detector: {e}")

        # 3. Initialize FaceNet
        try:
            self.facenet_model = DeepFace.build_model("Facenet512")
            print("Facenet512 model initialized.")
        except Exception as e:
            print(f"Error initializing Facenet512: {e}")

    # --- NEW METHOD: THE HOT RELOAD ---
    def load_brain(self):
        """Reads the .pkl file from the hard drive and overwrites RAM."""
        try:
            if os.path.exists(self.brain_file):
                with open(self.brain_file, "rb") as f:
                    new_brain = pickle.load(f)
                
                # Lock only for the exact microsecond we swap the pointer
                with self._brain_lock:
                    self.known_faces = new_brain
                print(f"[FaceRecognizer] Hot-Reloaded {len(self.known_faces)} known faces.")
            else:
                with self._brain_lock:
                    self.known_faces = {}
                print("[FaceRecognizer] Brain file not found. Cleared memory.")
        except Exception as e:
            print(f"[FaceRecognizer] Error reloading brain: {e}")



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

            # --- BUG C-3 FIX: Lock the dictionary iteration! ---
            with self._brain_lock:
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