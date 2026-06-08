import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.core.nova_logger import logger

class FaceDetector:
    """
    Refactored face detector as a callable class.
    Initializes MediaPipe FaceDetector and provides a detect method.
    """

    def __init__(self, model_path='blaze_face_short_range.tflite'):
        """
        Initialize the FaceDetector with the given model.
        """
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
        except Exception as e:
            logger.exception(f"Error initializing FaceDetector: {e}")
            self.detector = None

    def detect(self, mp_image):
        """
        Detect faces in the given MediaPipe image.
        Returns detection_result or None if error.
        """
        if self.detector is None:
            return None
        try:
            return self.detector.detect(mp_image)
        except Exception as e:
            logger.exception(f"Detection error: {e}")
            return None

    def close(self):
        """
        Close the detector.
        """
        if self.detector:
            self.detector.close()