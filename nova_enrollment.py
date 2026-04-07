import os
import threading
import time
from nova_face_detector import FaceDetector
from nova_audio import VoiceQueue
from nove_face_encoder import build_nova_brain
import re
import cv2
import mediapipe as mp

class EnrollmentManager:
    """Guided enrollment state machine that captures 5 pose frames without blocking."""

    STEPS = [
        ("Please look straight at the camera.", "straight"),
        ("Please turn your head slightly to the right.", "right"),
        ("Please turn your head slightly to the left.", "left"),
        ("Please look up a little.", "up"),
        ("Please look down a little.", "down"),
    ]

    def __init__(self, voice_queue, wait_seconds=5.0, db_folder="face_db"):
        self.voice_queue = voice_queue
        self.wait_seconds = wait_seconds
        self.db_folder = db_folder
        self.active = False
        self.name = None
        self.step_index = 0
        self.step_started = False
        self.step_saved = False
        self.step_start_time = 0.0
        self.detector = FaceDetector()
        self.last_warning_time = 0.0
        os.makedirs(self.db_folder, exist_ok=True)

    def _sanitize_name(self, raw_name):
        
        cleaned = raw_name.strip()
        cleaned = re.sub(r"[^a-zA-Z0-9 _-]", "", cleaned)
        return cleaned.title() if cleaned else None

    def start(self, name, current_time):
        sanitized = self._sanitize_name(name)
        if not sanitized:
            return False
        self.name = sanitized
        self.active = True
        self.step_index = 0
        self.step_started = False
        self.step_saved = False
        self.step_start_time = current_time
        self.voice_queue.speak(f"Starting enrollment for {self.name}.", VoiceQueue.PRIORITY_INFO)
        return True

    def update(self, frame, current_time):
        if not self.active: return

        # 1. Start the step and announce instructions
        if not self.step_started:
            instruction, _ = self.STEPS[self.step_index]
            # Use PRIORITY_SYSTEM so it doesn't get blocked
            self.voice_queue.speak(instruction, VoiceQueue.PRIORITY_SYSTEM) 
            self.step_start_time = current_time
            self.step_started = True
            self.step_saved = False
            return

        if not self.step_saved:
            # 2. VALIDATION GATE: Check if a face is actually there!

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = self.detector.detect(mp_image)

            # If no face is detected, reset the 3-second timer and warn the user
            if not detection_result or len(detection_result.detections) == 0:
                self.step_start_time = current_time # Reset timer!
                if current_time - self.last_warning_time > 3.0:
                    self.voice_queue.speak("I can't see your face clearly. Please face the camera.", VoiceQueue.PRIORITY_SYSTEM)
                    self.last_warning_time = current_time
                return

            # 3. If face is detected, proceed with the timer
            if current_time - self.step_start_time >= self.wait_seconds:
                self._save_frame(frame)
                self.step_saved = True
                self.step_index += 1

                if self.step_index >= len(self.STEPS):
                    self._complete()
                else:
                    self.step_started = False
    
    def cancel(self):
        """
        Abort any in-progress enrollment immediately.
        Safe to call whether or not enrollment is active.
        """
        if self.active:
            print(f"[EnrollmentManager] Enrollment for '{self.name}' cancelled.")
        self.active       = False
        self.name         = None
        self.step_index   = 0
        self.step_started = False
        self.step_saved   = False

    def _save_frame(self, frame):
        _, step_name = self.STEPS[self.step_index]
        safe_name = self.name.replace(" ", "_")
        filename = f"{safe_name}_{step_name}.jpg"
        path = os.path.join(self.db_folder, filename)
        cv2.imwrite(path, frame)
        print(f"Saved enrollment frame: {path}")

    def _complete(self):
        self.active = False
        self.voice_queue.speak(f"Enrollment for {self.name} complete.", VoiceQueue.PRIORITY_INFO)
        thread = threading.Thread(target=self._build_brain, args=(self.name,), daemon=True)
        thread.start()

    def _build_brain(self, name):
        try:
            self.voice_queue.speak("Building enrollment database.", VoiceQueue.PRIORITY_INFO)
            build_nova_brain(name)
            self.voice_queue.speak("Enrollment database updated.", VoiceQueue.PRIORITY_INFO)
        except Exception as e:
            print(f"Enrollment build error: {e}")
            self.voice_queue.speak("Enrollment failed.", VoiceQueue.PRIORITY_WARNING)