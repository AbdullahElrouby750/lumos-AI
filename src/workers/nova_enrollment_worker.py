import os
import queue
import re
import threading
import time
from typing import Any, Optional

from src.modules.nove_face_encoder import build_nova_brain
from src.modules.nova_pose_validator import PoseValidator
from src.network.nova_network_models import BaseEvent
from src.network.nova_server import LumosServer


class NovaEnrollmentWorker:
    """Isolated enrollment worker for head-pose guided face registration."""

    STEPS = [
        ("Please look straight at the camera.", "straight"),
        ("Please turn your head slightly to the right.", "right"),
        ("Please turn your head slightly to the left.", "left"),
        ("Please look up a little.", "up"),
        ("Please look down a little.", "down"),
    ]

    DB_FOLDER = "face_db"
    PRIORITY_SYSTEM = 3
    PRIORITY_FEEDBACK = 2
    PRIORITY_WARNING = 4
    PRIORITY_INFO = 6

    def __init__(self, server: LumosServer, wait_seconds: float = 5.0):
        self.server = server
        self.wait_seconds = wait_seconds
        self._frame_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._command_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()

        self.active = False
        self.target_name: Optional[str] = None
        self.step_index = 0
        self.step_started = False
        self.step_saved = False
        self.step_start_time = 0.0
        self.last_warning_time = 0.0

        self.validator = PoseValidator()
        os.makedirs(self.DB_FOLDER, exist_ok=True)

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="NovaEnrollmentWorker",
        )
        self._worker_thread.start()

    def start_session(self, target_name: str, on_complete=None) -> bool:
        sanitized = self._sanitize_name(target_name)
        if not sanitized:
            self._send_tts("Invalid enrollment name.", self.PRIORITY_SYSTEM)
            return False

        try:
            self._on_complete = on_complete
            self._command_queue.put_nowait(sanitized)
            return True
        except queue.Full:
            self._send_tts("Enrollment request dropped. Try again later.", self.PRIORITY_WARNING)
            return False

    def enqueue_frame(self, frame: Any) -> bool:
        try:
            self._frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)
        # --- BUG F-3 FIX: Check for zombie encoding threads ---
        if self._worker_thread.is_alive():
            print("[NovaEnrollmentWorker] CRITICAL WARNING: Thread still active after 5s timeout! A brain build is likely in progress.")
        # ------------------------------------------------------
        self.validator.close()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._consume_commands()

            if not self.active:
                time.sleep(0.1)
                continue

            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_frame(frame)
            finally:
                self._frame_queue.task_done()

    def _consume_commands(self) -> None:
        while True:
            try:
                target_name = self._command_queue.get_nowait()
            except queue.Empty:
                break

            if self.active:
                self._send_tts("Enrollment already in progress. Please wait.", self.PRIORITY_WARNING)
                self._command_queue.task_done()
                continue

            self._begin_session(target_name)
            self._command_queue.task_done()

    def _begin_session(self, target_name: str) -> None:
        self.target_name = target_name
        self.active = True
        self.step_index = 0
        self.step_started = False
        self.step_saved = False
        self.step_start_time = 0.0
        self.last_warning_time = 0.0
        self.validator.reset()
        self._send_tts(f"Starting enrollment for {self.target_name}.", self.PRIORITY_SYSTEM)

    def _process_frame(self, frame: Any) -> None:
        if not self.active or self.target_name is None:
            return

        instruction, step_name = self.STEPS[self.step_index]

        if not self.step_started:
            self._send_tts(instruction, self.PRIORITY_SYSTEM)
            self.step_start_time = time.time()
            self.step_started = True
            self.step_saved = False
            return

        if self.step_saved:
            return

        elapsed = time.time() - self.step_start_time
        if elapsed < self.wait_seconds:
            return

        is_valid, feedback = self.validator.validate_pose(frame, step_name)
        if is_valid:
            self._save_frame(frame, step_name)
            self._send_tts("Pose looks good!", self.PRIORITY_FEEDBACK)
            self.step_saved = True
            self.step_index += 1
            self.validator.reset()
            self._send_progress()

            if self.step_index >= len(self.STEPS):
                self._complete_session()
            else:
                self.step_started = False
                self.step_saved = False
        else:
            self.step_start_time = time.time()
            if feedback:
                self._send_tts(feedback, self.PRIORITY_SYSTEM)

    def _complete_session(self) -> None:
        self.active = False
        self._send_tts(f"Enrollment for {self.target_name} complete.", self.PRIORITY_FEEDBACK)
        self._send_tts("Building enrollment database.", self.PRIORITY_INFO)

        try:
            build_nova_brain(self.target_name)
            self._send_tts("Enrollment database updated.", self.PRIORITY_INFO)
            self._send_event("BRAIN_RELOAD_REQUEST", {"name": self.target_name})
        except Exception as exc:
            print(f"[NovaEnrollmentWorker] Brain build failed: {exc}")
            self._send_tts("Enrollment failed.", self.PRIORITY_SYSTEM)
        finally:
            temp_name = self.target_name  # Store before clearing
            self.target_name = None
            self._drain_queues()
            if self._on_complete:
                self._on_complete(temp_name)

    def _save_frame(self, frame: Any, step_name: str) -> None:
        safe_name = self.target_name.replace(" ", "_") if self.target_name else "unknown"
        filename = f"{safe_name}_{step_name}.jpg"
        path = os.path.join(self.DB_FOLDER, filename)
        try:
            import cv2
            cv2.imwrite(path, frame)
        except Exception as exc:
            print(f"[NovaEnrollmentWorker] Failed to save enrollment frame: {exc}")

    def _send_tts(self, text: str, priority: int) -> None:
        event = BaseEvent.create("TTS_AUDIO", {"text": text}, priority=priority)
        self.server.send_event(event)

    def _send_progress(self) -> None:
        percent = int((self.step_index / len(self.STEPS)) * 100)
        event = BaseEvent.create("ENROLLMENT_PROGRESS", {"percent": percent}, priority=self.PRIORITY_INFO)
        self.server.send_event(event)

    def _send_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = BaseEvent.create(event_type, payload, priority=self.PRIORITY_INFO)
        self.server.send_event(event)

    def _drain_queues(self) -> None:
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
            except queue.Empty:
                break

        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
                self._command_queue.task_done()
            except queue.Empty:
                break

    @staticmethod
    def _sanitize_name(raw_name: str) -> Optional[str]:
        cleaned = raw_name.strip()
        cleaned = re.sub(r"[^a-zA-Z0-9 _-]", "", cleaned)
        return cleaned.title() if cleaned else None
