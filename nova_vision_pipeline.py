import threading
import time
import cv2
import mediapipe as mp
from nova_face_detector import FaceDetector
from nove_face_rec import FaceRecognizer
from utils import CentroidTracker, is_in_collision_zone, is_bbox_expanding, draw_bounding_box, draw_text

class VisionPipeline:
    """Handles vision processing: detection, tracking, async recognition, and alerts."""

    def __init__(self, voice_queue):
        self.voice_queue = voice_queue
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.tracker = CentroidTracker(max_distance=200, max_disappeared=60)
        self.recognition_threads = {}
        self.recognition_results = {}
        self.temporal_votes = {} # NEW: {face_id: [guess1, guess2, guess3]}
        self.recognized_ids = set()
        self.previous_bboxes = {}
        self.alerted_approaching_ids = set()
        self.name_cooldowns = {}
        self.COOLDOWN_TIME = 30.0

    def process_frame(self, frame, current_time):
        """Process a single frame: detect, track, recognize, alert."""


        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = self.detector.detect(mp_image)

        detected_bboxes = []
        if detection_result and detection_result.detections:
            detected_bboxes = [d.bounding_box for d in detection_result.detections]

        tracked_ids_with_bboxes = self.tracker.update(detected_bboxes)

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        for face_id, bbox in tracked_ids_with_bboxes.items():
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            y_start = max(0, y)
            y_end = min(frame.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(frame.shape[1], x + w)

            if y_end > y_start and x_end > x_start:
                face_crop = frame[y_start:y_end, x_start:x_end].copy()
                # Only spawn a new thread if we haven't locked in a final result
                # AND if there isn't already a thread actively working on this face ID
                if face_id not in self.recognition_results and face_id not in self.recognition_threads:
                    
                    def recognize_worker(fid, crop):
                        try:
                            # 1. Get the heavy alignment guess
                            name = self.recognizer.recognize_face_crop(crop)
                            
                            # 2. Initialize the vote list if empty
                            if fid not in self.temporal_votes:
                                self.temporal_votes[fid] = []
                            
                            # 3. Append the new guess
                            self.temporal_votes[fid].append(name)
                            
                            # 4. THE CONSENSUS GATE: Check if we have 3 votes yet
                            if len(self.temporal_votes[fid]) >= 3:
                                votes = self.temporal_votes[fid][-3:]
                                final_name = max(set(votes), key=votes.count)
                                # Lock in the final result!
                                self.recognition_results[fid] = final_name
                                
                        except Exception as e:
                            print(f"Recognition thread error: {e}")
                            # Don't lock in "Unknown" on an error, let it try again on the next frame
                        finally:
                            # IMPORTANT: Clear the thread lock so the main loop can spawn 
                            # another thread for the next vote if we haven't reached 3 votes yet
                            self.recognition_threads.pop(fid, None)

                    # Spawn the thread and lock it so we don't spawn 30 threads in one second
                    thread = threading.Thread(
                        target=recognize_worker,
                        args=(face_id, face_crop),
                        daemon=True
                    )
                    self.recognition_threads[face_id] = thread
                    thread.start()

            if face_id in self.recognition_results:
                name = self.recognition_results[face_id]
                if name != "Unknown":
                    last_spoken_time = self.name_cooldowns.get(name, 0)
                    if current_time - last_spoken_time > self.COOLDOWN_TIME:
                        self.voice_queue.speak(
                            f"I see {name}",
                            self.voice_queue.PRIORITY_SOCIAL
                        )
                        self.name_cooldowns[name] = current_time
                    self.recognized_ids.add(face_id)
                    self.alerted_approaching_ids.discard(face_id)

                if name != "Unknown" and face_id in self.recognized_ids:
                    if is_in_collision_zone(bbox, frame_width, zone_percent=0.4):
                        prev_bbox = self.previous_bboxes.get(face_id)
                        if is_bbox_expanding(bbox, prev_bbox, threshold=10):
                            if face_id not in self.alerted_approaching_ids:
                                self.voice_queue.speak(
                                    f"{name} approaching",
                                    self.voice_queue.PRIORITY_WARNING
                                )
                                self.alerted_approaching_ids.add(face_id)
                    else:
                        self.alerted_approaching_ids.discard(face_id)

                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"ID:{face_id} {name}"
                else:
                    color = (255, 255, 0)
                    label = f"ID:{face_id} (recognizing...)"

                draw_bounding_box(frame, bbox, color=color, thickness=2)
                draw_text(frame, label, (bbox.origin_x, bbox.origin_y - 10), scale=0.8, color=color)

            self.previous_bboxes[face_id] = bbox

        current_tracked_ids = set(tracked_ids_with_bboxes.keys())
        for fid in list(self.recognized_ids):
            if fid not in current_tracked_ids:
                self.recognized_ids.discard(fid)
                self.previous_bboxes.pop(fid, None)
                self.recognition_results.pop(fid, None)
                self.recognition_threads.pop(fid, None)
                self.temporal_votes.pop(fid, None)
                self.alerted_approaching_ids.discard(fid)

        # Draw collision zone lines
        zone_left = int(frame_width * 0.3)
        zone_right = int(frame_width * 0.7)
        cv2.line(frame, (zone_left, 0), (zone_left, frame_height), (100, 100, 100), 1)
        cv2.line(frame, (zone_right, 0), (zone_right, frame_height), (100, 100, 100), 1)

        return frame

    def close(self):
        """Close detectors."""
        self.detector.close()
        self.recognizer.close()