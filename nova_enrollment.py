import os
import threading
import time

from nova_audio import VoiceQueue
from nove_face_encoder import build_nova_brain

class EnrollmentManager:
    """Guided enrollment state machine that captures 5 pose frames without blocking."""

    STEPS = [
        ("Please look straight at the camera.", "straight"),
        ("Please turn your head slightly to the right.", "right"),
        ("Please turn your head slightly to the left.", "left"),
        ("Please look up a little.", "up"),
        ("Please look down a little.", "down"),
    ]

    def __init__(self, voice_queue, wait_seconds=3.0, db_folder="face_db"):
        self.voice_queue = voice_queue
        self.wait_seconds = wait_seconds
        self.db_folder = db_folder
        self.active = False
        self.name = None
        self.step_index = 0
        self.step_started = False
        self.step_saved = False
        self.step_start_time = 0.0
        os.makedirs(self.db_folder, exist_ok=True)

    def _sanitize_name(self, raw_name):
        import re
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
        if not self.active:
            return

        if not self.step_started:
            prompt, _ = self.STEPS[self.step_index]
            self.voice_queue.speak(prompt, VoiceQueue.PRIORITY_INFO)
            self.step_start_time = current_time
            self.step_started = True
            self.step_saved = False
            return

        if not self.step_saved and current_time - self.step_start_time >= self.wait_seconds:
            self._save_frame(frame)
            self.step_saved = True
            self.step_index += 1

            if self.step_index >= len(self.STEPS):
                self._complete()
            else:
                self.step_started = False

    def _save_frame(self, frame):
        _, step_name = self.STEPS[self.step_index]
        safe_name = self.name.replace(" ", "_")
        filename = f"{safe_name}_{step_name}.jpg"
        path = os.path.join(self.db_folder, filename)
        import cv2
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