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
        self.bbox_history_buffer = {} # NEW: {face_id: [bbox1, bbox2, ...]}
        self.alerted_approaching_ids = {} # CHANGED: Now a Dict for timestamp locks
        self.name_cooldowns = {}
        self.unknown_timers = {} # Tracks when a face was labeled "Unknown"
        self.COOLDOWN_TIME = 120.0
        
    def hot_reload(self):
        """Forces the recognizer to reload the brain and wipes the pipeline's short-term memory."""
        print("[VisionPipeline] Hot-reload triggered. Wiping short-term memory...")
        self.recognizer.load_brain() # Reload the hard drive
        
        # Re-assign dictionaries to instantly wipe the cache (Thread-safe in Python)
        self.recognition_results = {}
        self.temporal_votes = {}
        self.recognized_ids = set()
        self.unknown_timers = {}
        self.name_cooldowns = {}
        self.alerted_approaching_ids = {}

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
                
                # --- NEW: EXPIRATION GATE FOR UNKNOWNS ---
                # If they've been 'Unknown' for more than 10 seconds, wipe their memory so we try again!
                if face_id in self.recognition_results and self.recognition_results[face_id] == "Unknown":
                    if current_time - self.unknown_timers.get(face_id, 0) > 10.0:
                        self.recognition_results.pop(face_id, None)
                        self.temporal_votes.pop(face_id, None)
                        self.unknown_timers.pop(face_id, None) # <--- YOUR FIX: Clean the timer!
                        print(f"DEBUG: 10 seconds passed. Re-evaluating Unknown Face ID: {face_id}")
                # ----------------------------------------
                
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
                                
                                # NEW: Start the 10-second expiration clock if they are Unknown
                                if final_name == "Unknown":
                                    self.unknown_timers[fid] = time.time()
                                
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
                            f"hi, I see {name}",
                            self.voice_queue.PRIORITY_SOCIAL
                        )
                        self.name_cooldowns[name] = current_time
                    self.recognized_ids.add(face_id)
                    

                if name != "Unknown" and face_id in self.recognized_ids:
                    # 1. ALWAYS update the Jitter Buffer for recognized faces
                    if face_id not in self.bbox_history_buffer:
                        self.bbox_history_buffer[face_id] = []
                    self.bbox_history_buffer[face_id].append(bbox)

                    # Keep only the last 5 frames for the moving average
                    if len(self.bbox_history_buffer[face_id]) > 5:
                        self.bbox_history_buffer[face_id].pop(0)

                    # 2. Check the Collision Zone & Cooldown
                    if is_in_collision_zone(bbox, frame_width, zone_percent=0.4):
                        # Get the last time we warned about this person (default to 0)
                        last_alert_time = self.alerted_approaching_ids.get(face_id, 0)

                        # 5-SECOND LOCK: Only calculate expansion if 5 seconds have passed
                        if current_time - last_alert_time > 5.0:
                            if is_bbox_expanding(bbox, self.bbox_history_buffer[face_id], threshold=20):
                                # FIRE WARNING!
                                self.voice_queue.speak(f"hello, {name} approaching", self.voice_queue.PRIORITY_WARNING)
                                # Lock it with the current timestamp!
                                self.alerted_approaching_ids[face_id] = current_time 
                                self.bbox_history_buffer[face_id] = [] # Clear the jitter buffer after a warning to prevent multiple warnings from the same approach
                else:
                    self.alerted_approaching_ids.pop(face_id, None)

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
                self.bbox_history_buffer.pop(fid, None) # NEW: Clean up jitter buffer
                self.alerted_approaching_ids.pop(fid, None) # CHANGED: Now uses .pop since it's a dict
                self.unknown_timers.pop(fid, None) # NEW: Clean up unknown timer

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