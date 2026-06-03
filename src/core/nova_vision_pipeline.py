import queue
import threading
import time
import cv2
import mediapipe as mp
from src.modules.nova_face_detector import FaceDetector
from src.modules.nove_face_rec import FaceRecognizer
from src.core.utils import (
    CentroidTracker, is_in_collision_zone, is_bbox_expanding,
    draw_bounding_box, draw_text,
)
from src.core.nova_commands import (
    INTENT_CONFIRM, INTENT_DENY, INTENT_SCENE, INTENT_TEXT, INTENT_ENROLL, INTENT_FORGET, INTENT_QUIT
)
import difflib  # <--- NEW
from src.modules.nove_forget import forget_person  # <--- NEW

# ── Tuning constants ──────────────────────────────────────────────────────────
# A face must be tracked for this many consecutive frames before we spend any
# GPU/CPU cycles running DeepFace on it.  At 30 FPS a pan produces ghost IDs
# that survive only 1-3 frames; this gate kills them all before they queue.
_STABILITY_FRAMES_REQUIRED = 10

# Hard ceiling on simultaneous DeepFace threads. 
# TensorFlow WILL crash if this is higher than 1 during rapid movement.
_MAX_RECOGNITION_THREADS = 1
# ─────────────────────────────────────────────────────────────────────────────


class VisionPipeline:
    """
    Handles vision processing: detection, tracking, async recognition, alerts.

    Thread-safety model
    -------------------
    All state dictionaries are only read and written from the *main camera
    thread* (process_frame).  The recognize_worker background thread is the
    sole exception: it appends to self.temporal_votes and writes to
    self.recognition_results / self.unknown_timers.  Every such cross-thread
    access is serialised through self._state_lock.

    self.recognition_threads is also protected by self._state_lock because
    both the main thread (reads for the spawn-gate check, writes on spawn) and
    the worker thread (deletes itself in finally) touch it concurrently.
    """

    def __init__(self, voice_queue, server=None, ai_worker=None, yolo_worker=None, enrollment_worker=None):
        self.voice_queue = voice_queue
        self.server = server
        self.ai_worker = ai_worker
        self.yolo_worker = yolo_worker
        self.enrollment_worker = enrollment_worker
        self.is_enrolling = False
        # --- BUG E-1 FIX: Add state variable ---
        self.pending_forget_name = None 
        # ---------------------------------------
        # --- BUG F-1 FIX (Part 1): Add the quit signal flag ---
        self.quit_requested = False
        # ------------------------------------------------------
        self._command_inbox: queue.Queue[dict] = queue.Queue()
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.tracker = CentroidTracker(max_distance=200, max_disappeared=60)

        # ── Per-face state (main-thread only, except where noted) ─────────────
        self.recognition_threads = {}   # {face_id: Thread}  – also touched by workers
        self.recognition_results = {}   # {face_id: str}      – also touched by workers
        self.temporal_votes      = {}   # {face_id: [str]}    – also touched by workers
        self.unknown_timers      = {}   # {face_id: float}    – also touched by workers
        self.recognized_ids      = set()
        self.previous_bboxes     = {}
        self.bbox_history_buffer = {}   # {face_id: [bbox]}
        self.alerted_approaching_ids = {}  # {face_id: float}  timestamp lock
        self.name_cooldowns      = {}   # {name: float}

        # ── BUG 1 FIX: stability counter ─────────────────────────────────────
        # Counts how many consecutive frames each face_id has been tracked.
        # A thread is only allowed to spawn once this reaches _STABILITY_FRAMES_REQUIRED.
        self._stability_counter  = {}   # {face_id: int}

        # ── BUG 1 FIX: single lock that protects all cross-thread dict access ─
        self._state_lock = threading.Lock()

        # ── BUG 1 FIX: thread-pool semaphore ─────────────────────────────────
        # Blocks thread spawning when _MAX_RECOGNITION_THREADS are already live.
        self._thread_semaphore = threading.BoundedSemaphore(_MAX_RECOGNITION_THREADS)

        self.COOLDOWN_TIME = 120.0
        self.is_paused     = False

    # ── Public helpers ────────────────────────────────────────────────────────

    def on_command_received(self, command: dict) -> bool:
        """
        Acts as the direct callback for LumosServer. 
        Drops the WebSocket command into the CEO's inbox safely.
        """
        try:
            self._command_inbox.put_nowait(command)
            return True
        except queue.Full:
            print(f"[VisionPipeline] Command inbox full, dropping command: {command}")
            return False

    def process_inbox(self):
        """The Switchboard Router: Process all pending commands without waiting."""
        while True:
            try:
                command = self._command_inbox.get_nowait()
            except queue.Empty:
                break

            try:
                intent = command.get("intent")
                print(f"[VisionPipeline] CEO routing intent: {intent}")

                # 1. Route to AI Worker
                if intent in [INTENT_SCENE, INTENT_TEXT]:
                    if self.ai_worker:
                        # Drop it into the AI's waiting room
                        # --- BUG D-1 FIX: Use the public API ---
                        success = self.ai_worker.enqueue_command(command)
                        if not success:
                            print("[VisionPipeline] Warning: AI Worker queue is full. Command dropped.")
                        # ---------------------------------------
                    else:
                        print("[VisionPipeline] Warning: AI Worker not initialized.")

                # 2. Route to Enrollment
                elif intent == INTENT_ENROLL:
                    target_name = command.get("target_name")
                    if self.enrollment_worker and target_name:
                        with self._state_lock:
                            self.is_enrolling = True
                        
                        # --- BUG C-5 FIX: Removed the dead try/except block ---
                        self.enrollment_worker.start_session(
                            target_name,
                            on_complete=self._on_enrollment_done,
                        )
                        # ------------------------------------------------------
                    else:
                        print("[VisionPipeline] Enrollment request missing target_name or worker not configured.")

                # 3. Route to Forget (Will wire up later)
                elif intent == INTENT_FORGET:
                    target_name = command.get("target_name", "").strip()
                    
                    with self._state_lock:
                        known_names = list(self.recognizer.known_faces.keys())

                    # If they didn't specify a name, read out the list
                    if not target_name:
                        if not known_names:
                            self.voice_queue.speak("I don't know anyone yet.", self.voice_queue.PRIORITY_SYSTEM)
                        else:
                            name_list = ", ".join(known_names)
                            self.voice_queue.speak(f"Who should I forget? I know: {name_list}", self.voice_queue.PRIORITY_SYSTEM)
                        continue

                    # Fuzzy match (If user says "Ruby", find "Rouby")
                    best_matches = difflib.get_close_matches(target_name, known_names, n=1, cutoff=0.6)

                    if best_matches:
                        matched_name = best_matches[0]
                        with self._state_lock:
                            self.pending_forget_name = matched_name
                        self.voice_queue.speak(f"Are you sure you want to forget {matched_name}? Say yes to confirm.", self.voice_queue.PRIORITY_FEEDBACK)
                    else:
                        self.voice_queue.speak(f"I couldn't find {target_name}. Please try again.", self.voice_queue.PRIORITY_FEEDBACK)

                # 4. Route Confirm / Deny (For the Forget Protocol)
                elif intent == INTENT_CONFIRM:
                    with self._state_lock:
                        target = self.pending_forget_name
                        
                    if target:
                        # 1. Delete from hard drive
                        success = forget_person(target)
                        if success:
                            self.voice_queue.speak(f"I have forgotten {target}.", self.voice_queue.PRIORITY_SYSTEM)
                            # 2. Hot-reload the RAM to wipe short-term memory
                            self.hot_reload()
                        else:
                            self.voice_queue.speak(f"Error trying to forget {target}.", self.voice_queue.PRIORITY_WARNING)
                        
                        # 3. Clear the pending state
                        with self._state_lock:
                            self.pending_forget_name = None

                elif intent == INTENT_DENY:
                    with self._state_lock:
                        if self.pending_forget_name:
                            self.pending_forget_name = None
                            self.voice_queue.speak("Forget request cancelled.", self.voice_queue.PRIORITY_SYSTEM)

                # 4. Global Shutdown
                elif intent == INTENT_QUIT:
                    print("[VisionPipeline] Quit command received. Signaling main loop...")
                    # --- BUG F-1 FIX (Part 2): Press the button ---
                    self.quit_requested = True
                    # ----------------------------------------------

            except Exception as e:
                print(f"[VisionPipeline] Command routing error: {e}")
            finally:
                self._command_inbox.task_done()

    def _on_enrollment_done(self, target_name: str):
        print(f"[VisionPipeline] Enrollment completed for {target_name}. Reloading brain...")
        self.hot_reload()
        with self._state_lock:
            self.is_enrolling = False

    def hot_reload(self):
        """Reload brain from disk and wipe short-term memory atomically."""
        print("[VisionPipeline] Hot-reload triggered. Wiping short-term memory...")
        self.recognizer.load_brain()

        with self._state_lock:
            self.recognition_results     = {}
            self.temporal_votes          = {}
            self.recognized_ids          = set()
            self.unknown_timers          = {}
            self.name_cooldowns          = {}
            self.alerted_approaching_ids = {}
            self._stability_counter      = {}
            # NOTE: we intentionally leave recognition_threads alone.
            # Any already-running threads are holding the semaphore; they will
            # release it in their finally block and remove themselves safely.

    # ── Main per-frame entry point ────────────────────────────────────────────

    def process_frame(self, frame, current_time):
        """Process a single frame: detect, track, recognise, alert."""

        self.process_inbox()
        if self.yolo_worker:
            self.yolo_worker.enqueue_frame(frame.copy())
        
        # --- BUG C-4 FIX: Lock the read check! ---
        with self._state_lock:
            currently_enrolling = self.is_enrolling
        
        if currently_enrolling and self.enrollment_worker:
            self.enrollment_worker.enqueue_frame(frame.copy())
            cv2.putText(
                frame,
                "ENROLLMENT IN PROGRESS",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            return frame

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        detection_result = self.detector.detect(mp_image)

        detected_bboxes = []
        if detection_result and detection_result.detections:
            detected_bboxes = [d.bounding_box for d in detection_result.detections]

        tracked_ids_with_bboxes = self.tracker.update(detected_bboxes)
        current_tracked_ids = set(tracked_ids_with_bboxes.keys())

        frame_width  = frame.shape[1]
        frame_height = frame.shape[0]

        # ── Per-face processing ───────────────────────────────────────────────
        for face_id, bbox in tracked_ids_with_bboxes.items():
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            y_start = max(0, y)
            y_end   = min(frame.shape[0], y + h)
            x_start = max(0, x)
            x_end   = min(frame.shape[1], x + w)

            if y_end > y_start and x_end > x_start:
                face_crop = frame[y_start:y_end, x_start:x_end].copy()
                
                # --- NEW: SAFE CROP CHECK ---
                # Prevent DeepFace from crashing on tiny or edge-screen crops
                if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                    continue
                # ----------------------------

                # ── BUG 1 FIX: increment stability counter ────────────────────
                self._stability_counter[face_id] = (
                    self._stability_counter.get(face_id, 0) + 1
                )
                is_stable = (
                    self._stability_counter[face_id] >= _STABILITY_FRAMES_REQUIRED
                )
                # ──────────────────────────────────────────────────────────────

                # Unknown expiration gate (read under lock because workers write these)
                with self._state_lock:
                    result_for_id = self.recognition_results.get(face_id)
                    unknown_since = self.unknown_timers.get(face_id, 0)

                if result_for_id == "Unknown":
                    if current_time - unknown_since > 10.0:
                        with self._state_lock:
                            self.recognition_results.pop(face_id, None)
                            self.temporal_votes.pop(face_id, None)
                            self.unknown_timers.pop(face_id, None)
                        self._stability_counter[face_id] = 0  # force re-stabilisation
                        print(
                            f"[VisionPipeline] Re-evaluating Unknown ID {face_id} "
                            f"after 10 s expiry."
                        )

                # ── BUG 1 FIX: spawn gate (stability + pool + lock) ───────────
                with self._state_lock:
                    has_result  = face_id in self.recognition_results
                    has_thread  = face_id in self.recognition_threads

                can_spawn = (
                    not self.is_paused
                    and is_stable
                    and not has_result
                    and not has_thread
                    and self._thread_semaphore.acquire(blocking=False)  # non-blocking
                )

                if can_spawn:
                    # Capture loop variable safely
                    def _make_worker(fid, crop):
                        def recognize_worker():
                            try:
                                with self._state_lock:
                                    # --- BUG A-2 FIX: THE ZOMBIE GATE ---
                                    # If the face vanished while DeepFace was thinking, 
                                    # the main thread deleted it from _stability_counter.
                                    # Abort the write so we don't leak memory.
                                    if fid not in self._stability_counter:
                                        return
                                    
                                    if fid not in self.temporal_votes:
                                        self.temporal_votes[fid] = []
                                        
                                    self.temporal_votes[fid].append(name)
                                    votes_snapshot = self.temporal_votes[fid][-3:]

                                    # We also combined the locks here for better performance!
                                    if len(self.temporal_votes[fid]) >= 3:
                                        final_name = max(
                                            set(votes_snapshot),
                                            key=votes_snapshot.count,
                                        )
                                        self.recognition_results[fid] = final_name
                                        if final_name == "Unknown":
                                            self.unknown_timers[fid] = time.time()

                            except Exception as e:
                                print(f"[VisionPipeline] Recognition thread error: {e}")

                            finally:
                                # Always release semaphore slot and remove self
                                self._thread_semaphore.release()
                                with self._state_lock:
                                    self.recognition_threads.pop(fid, None)

                        return recognize_worker

                    thread = threading.Thread(
                        target=_make_worker(face_id, face_crop),
                        daemon=True,
                    )
                    with self._state_lock:
                        self.recognition_threads[face_id] = thread
                    thread.start()

                elif not can_spawn and self._thread_semaphore._value == 0:
                    # Pool is full: we acquired nothing, no release needed.
                    # (If can_spawn was False for other reasons, acquire was never called.)
                    pass
                # ──────────────────────────────────────────────────────────────

            # ── Read final result (safe snapshot under lock) ──────────────────
            with self._state_lock:
                name = self.recognition_results.get(face_id)

            # --- NEW: ALWAYS DRAW BOXES ---
            # Default to a temporary state if the thread hasn't finished or is paused
            display_name = name if name is not None else "Analyzing..."

            if display_name != "Unknown" and display_name != "Analyzing...":
                with self._state_lock:
                    last_spoken = self.name_cooldowns.get(display_name, 0)
                    if current_time - last_spoken > self.COOLDOWN_TIME:
                        self.voice_queue.speak(
                            f"hi, I see {display_name}",
                            self.voice_queue.PRIORITY_SOCIAL,
                        )
                        self.name_cooldowns[display_name] = current_time
                    self.recognized_ids.add(face_id)

            if display_name != "Unknown" and display_name != "Analyzing..." and face_id in self.recognized_ids:
                # Jitter buffer update
                if face_id not in self.bbox_history_buffer:
                    self.bbox_history_buffer[face_id] = []
                self.bbox_history_buffer[face_id].append(bbox)
                if len(self.bbox_history_buffer[face_id]) > 5:
                    self.bbox_history_buffer[face_id].pop(0)

                # Collision zone + approaching alert
                if is_in_collision_zone(bbox, frame_width, zone_percent=0.4):
                    last_alert = self.alerted_approaching_ids.get(face_id, 0)
                    if current_time - last_alert > 5.0:
                        if is_bbox_expanding(bbox, self.bbox_history_buffer[face_id], threshold=20):
                            self.voice_queue.speak(
                                f"hello, {display_name} approaching",
                                self.voice_queue.PRIORITY_WARNING,
                            )
                            self.alerted_approaching_ids[face_id] = current_time
                            self.bbox_history_buffer[face_id] = []
                else:
                    self.alerted_approaching_ids.pop(face_id, None)

            # Draw (This now runs for EVERY tracked face)
            if display_name not in ["Unknown", "Analyzing..."]:
                color = (0, 255, 0)
                label = f"ID:{face_id} {display_name}"
            else:
                color = (255, 255, 0)
                label = f"ID:{face_id} ({display_name})"

            draw_bounding_box(frame, bbox, color=color, thickness=2)
            draw_text(
                frame, label,
                (bbox.origin_x, bbox.origin_y - 10),
                scale=0.8, color=color,
            )
            # ---------------------------------------------

            self.previous_bboxes[face_id] = bbox

        # ── BUG 1 FIX: full cleanup (covers ALL tracked dicts, not just recognized_ids)
        all_known_ids = set(self._stability_counter.keys())
        vanished_ids  = all_known_ids - current_tracked_ids

        for fid in vanished_ids:
            self._stability_counter.pop(fid, None)
            self.previous_bboxes.pop(fid, None)
            self.bbox_history_buffer.pop(fid, None)
            self.alerted_approaching_ids.pop(fid, None)
            with self._state_lock:
                self.recognized_ids.discard(fid)
                self.recognition_results.pop(fid, None)
                self.temporal_votes.pop(fid, None)
                self.unknown_timers.pop(fid, None)
                # Do NOT pop recognition_threads here: the worker thread is
                # still live and will pop itself in its finally block.
                # Removing it here would leak the semaphore slot.
        # ──────────────────────────────────────────────────────────────────────

        # Draw collision zone guides
        zone_left  = int(frame_width * 0.3)
        zone_right = int(frame_width * 0.7)
        cv2.line(frame, (zone_left,  0), (zone_left,  frame_height), (100, 100, 100), 1)
        cv2.line(frame, (zone_right, 0), (zone_right, frame_height), (100, 100, 100), 1)

        return frame

    def close(self):
        """Close detectors."""
        self.detector.close()
        self.recognizer.close()