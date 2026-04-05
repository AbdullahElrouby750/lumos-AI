import cv2
import mediapipe as mp
import json
import os
import re
import socket
import threading
import time
from queue import Empty, Queue

from nova_face_detector import FaceDetector
from nove_face_rec import FaceRecognizer
from nove_face_encoder import build_nova_brain
from nove_forget import forget_person, get_names_from_PK
from nova_audio import VoiceQueue
from nova_commands import parse_intent, INTENT_ENROLL, INTENT_FORGET, INTENT_NONE
from utils import (
    draw_bounding_box, draw_text, calculate_fps,
    CentroidTracker, is_in_collision_zone, is_bbox_expanding
)

COMMAND_HOST = "127.0.0.1"
COMMAND_PORT = 55555
ENROLLMENT_WAIT_SECONDS = 3.0


class CommandListener(threading.Thread):
    """Listens for local JSON voice commands on UDP and pushes them into a queue."""

    def __init__(self, command_queue, host=COMMAND_HOST, port=COMMAND_PORT):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.host = host
        self.port = port
        self.running = True
        self.sock = None

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(1.0)
            print(f"Command listener bound to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start command listener: {e}")
            self.running = False
            return

        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                raw_text = payload.get("raw_text", "")
                command = parse_intent(raw_text)
                if command["intent"] != INTENT_NONE:
                    command["raw_text"] = raw_text
                    self.command_queue.put(command)
                    print(f"Received voice command: {command}")
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print("Received invalid JSON command payload.")
            except Exception as e:
                print(f"Command listener error: {e}")

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


class EnrollmentManager:
    """Guided enrollment state machine that captures 5 pose frames without blocking."""

    STEPS = [
        ("Please look straight at the camera.", "straight"),
        ("Please turn your head slightly to the right.", "right"),
        ("Please turn your head slightly to the left.", "left"),
        ("Please look up a little.", "up"),
        ("Please look down a little.", "down"),
    ]

    def __init__(self, voice_queue, wait_seconds=ENROLLMENT_WAIT_SECONDS, db_folder="face_db"):
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


def process_command(command, enrollment_manager, voice_queue):
    intent = command.get("intent")
    target_name = command.get("target_name")

    if intent == INTENT_ENROLL:
        if not target_name:
            voice_queue.speak("Please say the name to enroll.", VoiceQueue.PRIORITY_WARNING)
            return
        if enrollment_manager.active:
            voice_queue.speak("Enrollment is already in progress.", VoiceQueue.PRIORITY_WARNING)
            return
        enrollment_manager.start(target_name, time.time())
        return

    if intent == INTENT_FORGET:
        if not target_name:
            voice_queue.speak("Please say the name to forget.", VoiceQueue.PRIORITY_WARNING)
            return

        names_stored = get_names_from_PK()
        if not names_stored:
            voice_queue.speak("No stored people found.", VoiceQueue.PRIORITY_WARNING)
            return

        result = forget_person(target_name, names_stored)
        voice_queue.speak(result, VoiceQueue.PRIORITY_WARNING)
        return

    # Ignore INTENT_NONE


def main():
    """Main CEO script for Lumos with voice command input and guided enrollment."""
    voice_queue = VoiceQueue()
    voice_queue.speak("Lumos starting", VoiceQueue.PRIORITY_INFO)

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    if detector.detector is None or recognizer.facenet_model is None:
        print("Failed to initialize detector or recognizer. Exiting.")
        voice_queue.speak("Initialization failed", VoiceQueue.PRIORITY_DANGER)
        detector.close()
        recognizer.close()
        voice_queue.stop()
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Camera not accessible")
    except Exception as e:
        print(f"Error opening camera: {e}")
        voice_queue.speak("Camera error", VoiceQueue.PRIORITY_DANGER)
        detector.close()
        recognizer.close()
        voice_queue.stop()
        return

    command_queue = Queue()
    command_listener = CommandListener(command_queue)
    command_listener.start()

    enrollment_manager = EnrollmentManager(voice_queue)

    name_cooldowns = {}
    COOLDOWN_TIME = 30.0
    tracker = CentroidTracker(max_distance=200, max_disappeared=60)

    last_time = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    recognition_threads = {}
    recognition_results = {}
    recognized_ids = set()
    previous_bboxes = {}
    alerted_approaching_ids = set()

    print("Lumos: Ready for voice commands. Say 'enroll' or 'forget'.")
    print(f"Listening for commands on {COMMAND_HOST}:{COMMAND_PORT}")

    try:
        while cap.isOpened():
            current_time = time.time()
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            enrollment_manager.update(frame, current_time)

            try:
                while True:
                    command = command_queue.get_nowait()
                    process_command(command, enrollment_manager, voice_queue)
            except Empty:
                pass

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(mp_image)

            detected_bboxes = []
            if detection_result and detection_result.detections:
                detected_bboxes = [d.bounding_box for d in detection_result.detections]

            tracked_ids_with_bboxes = tracker.update(detected_bboxes)

            for face_id, bbox in tracked_ids_with_bboxes.items():
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                y_start = max(0, y)
                y_end = min(frame.shape[0], y + h)
                x_start = max(0, x)
                x_end = min(frame.shape[1], x + w)

                if y_end > y_start and x_end > x_start:
                    face_crop = frame[y_start:y_end, x_start:x_end].copy()
                    if face_id not in recognition_results and face_id not in recognition_threads:
                        def recognize_worker(fid, crop):
                            try:
                                name = recognizer.recognize_face_crop(crop)
                                recognition_results[fid] = name
                            except Exception as e:
                                print(f"Recognition thread error: {e}")
                                recognition_results[fid] = "Unknown"

                        thread = threading.Thread(
                            target=recognize_worker,
                            args=(face_id, face_crop),
                            daemon=True
                        )
                        thread.start()
                        recognition_threads[face_id] = thread

                if face_id in recognition_results:
                    name = recognition_results[face_id]
                    if name != "Unknown":
                        last_spoken_time = name_cooldowns.get(name, 0)
                        if current_time - last_spoken_time > COOLDOWN_TIME:
                            voice_queue.speak(
                                f"I see {name}",
                                VoiceQueue.PRIORITY_SOCIAL
                            )
                            name_cooldowns[name] = current_time
                        recognized_ids.add(face_id)
                        alerted_approaching_ids.discard(face_id)

                    if name != "Unknown" and face_id in recognized_ids:
                        if is_in_collision_zone(bbox, frame_width, zone_percent=0.4):
                            prev_bbox = previous_bboxes.get(face_id)
                            if is_bbox_expanding(bbox, prev_bbox, threshold=10):
                                if face_id not in alerted_approaching_ids:
                                    voice_queue.speak(
                                        f"{name} approaching",
                                        VoiceQueue.PRIORITY_DANGER
                                    )
                                    alerted_approaching_ids.add(face_id)
                        else:
                            alerted_approaching_ids.discard(face_id)

                    if name != "Unknown":
                        color = (0, 255, 0)
                        label = f"ID:{face_id} {name}"
                    else:
                        color = (255, 255, 0)
                        label = f"ID:{face_id} (recognizing...)"

                    draw_bounding_box(frame, bbox, color=color, thickness=2)
                    draw_text(frame, label, (bbox.origin_x, bbox.origin_y - 10), scale=0.8, color=color)

                previous_bboxes[face_id] = bbox

            current_tracked_ids = set(tracked_ids_with_bboxes.keys())
            for fid in list(recognized_ids):
                if fid not in current_tracked_ids:
                    recognized_ids.discard(fid)
                    previous_bboxes.pop(fid, None)
                    recognition_results.pop(fid, None)
                    recognition_threads.pop(fid, None)
                    alerted_approaching_ids.discard(fid)

            zone_left = int(frame_width * 0.3)
            zone_right = int(frame_width * 0.7)
            cv2.line(frame, (zone_left, 0), (zone_left, frame_height), (100, 100, 100), 1)
            cv2.line(frame, (zone_right, 0), (zone_right, frame_height), (100, 100, 100), 1)

            fps = calculate_fps(current_time, last_time)
            last_time = current_time
            draw_text(frame, f"FPS: {int(fps)}", (20, 50), scale=1.0, color=(255, 0, 0))

            cv2.imshow("Lumos Face Detection - Spam Filter Active", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        command_listener.stop()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        recognizer.close()

        voice_queue.speak("Lumos shutting down", VoiceQueue.PRIORITY_INFO)
        while voice_queue.get_queue_size() > 0:
            time.sleep(0.5)
        time.sleep(2)
        voice_queue.stop()


if __name__ == "__main__":
    main()
