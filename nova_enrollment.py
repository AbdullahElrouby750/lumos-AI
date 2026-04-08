import os
import threading
import time
from nova_audio import VoiceQueue
from nove_face_encoder import build_nova_brain
from nova_pose_validator import PoseValidator
import re
import cv2

class EnrollmentManager:
    """Guided enrollment state machine that captures 5 pose frames without blocking."""

    STEPS = [
        ("Please look straight at the camera.", "straight"),
        ("Please turn your head slightly to the right.", "right"),
        ("Please turn your head slightly to the left.", "left"),
        ("Please look up a little.", "up"),
        ("Please look down a little.", "down"),
    ]

    def __init__(self, voice_queue, vision_pipeline=None, wait_seconds=5.0, db_folder="face_db"):
        self.voice_queue = voice_queue
        self.vision_pipeline = vision_pipeline # NEW
        self.wait_seconds = wait_seconds
        self.db_folder = db_folder
        self.active = False
        self.name = None
        self.step_index = 0
        self.step_started = False
        self.step_saved = False
        self.step_start_time = 0.0
        self.validator = PoseValidator()
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
            # 2. VALIDATION GATE: Check pose at timer expiration
            if current_time - self.step_start_time >= self.wait_seconds:
                _, step_name = self.STEPS[self.step_index]
                is_valid, feedback = self.validator.validate_pose(frame, step_name)
                if is_valid:
                    self._save_frame(frame)
                    self.voice_queue.speak("Pose looks good!", VoiceQueue.PRIORITY_FEEDBACK)
                    self.step_saved = True
                    self.step_index += 1
                    if self.step_index >= len(self.STEPS):
                        self._complete()
                    else:
                        self.step_started = False
                else:
                    self.step_start_time = current_time  # Reset timer
                    self.voice_queue.speak(feedback, VoiceQueue.PRIORITY_SYSTEM)
    
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
            
            # NEW: Pause the vision pipeline to prevent thread collisions
            if self.vision_pipeline:
                self.vision_pipeline.is_paused = True
            
            build_nova_brain(name)

            # NEW: Trigger the hot-reload
            if self.vision_pipeline:
                self.vision_pipeline.hot_reload()

            self.voice_queue.speak("Enrollment database updated.", VoiceQueue.PRIORITY_INFO)
        except Exception as e:
            print(f"Enrollment build error: {e}")
            if self.vision_pipeline:
                self.vision_pipeline.is_paused = False
            self.voice_queue.speak("Enrollment failed.", VoiceQueue.PRIORITY_WARNING)