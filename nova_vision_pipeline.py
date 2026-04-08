import threading
import time
import cv2
import mediapipe as mp
from nova_face_detector import FaceDetector
from nove_face_rec import FaceRecognizer
from utils import (
    CentroidTracker, is_in_collision_zone, is_bbox_expanding,
    draw_bounding_box, draw_text,
)

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

    def __init__(self, voice_queue):
        self.voice_queue = voice_queue
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
                                name = self.recognizer.recognize_face_crop(crop)

                                with self._state_lock:
                                    if fid not in self.temporal_votes:
                                        self.temporal_votes[fid] = []
                                    self.temporal_votes[fid].append(name)
                                    vote_count = len(self.temporal_votes[fid])
                                    votes_snapshot = self.temporal_votes[fid][-3:]

                                if vote_count >= 3:
                                    final_name = max(
                                        set(votes_snapshot),
                                        key=votes_snapshot.count,
                                    )
                                    with self._state_lock:
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

            # ── Read final result (safe snapshot under lock) ──────────────────
            with self._state_lock:
                name = self.recognition_results.get(face_id)

            # --- NEW: ALWAYS DRAW BOXES ---
            # Default to a temporary state if the thread hasn't finished or is paused
            display_name = name if name is not None else "Analyzing..."

            if display_name != "Unknown" and display_name != "Analyzing...":
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
            self.recognized_ids.discard(fid)
            with self._state_lock:
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