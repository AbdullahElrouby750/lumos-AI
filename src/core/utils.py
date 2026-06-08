import cv2
import time
import math
import difflib
import random
from src.core.nova_logger import logger
from src.modules.nove_forget import forget_person, get_names_from_PK
from src.core.nova_commands import INTENT_ENROLL, INTENT_FORGET, INTENT_QUIT, INTENT_CANCEL, INTENT_CONFIRM, INTENT_DENY, INTENT_WAKE_ONLY

# Constants (preserved from existing code)
KNOWN_WIDTH = 14.0  # cm
FOCAL_LENGTH = 678.57  # Preserved calibrated value
# Global state for Forget confirmation
pending_forget_name = None

def calculate_distance(face_width_pixels):
    """
    Calculate the distance to a face in meters using the calibrated focal length.
    Preserves existing math formula.
    """
    if face_width_pixels <= 0:
        return float('inf')  # Avoid division by zero
    distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / face_width_pixels
    return distance_cm / 100.0  # Convert to meters

def draw_bounding_box(frame, bbox, color=(0, 255, 255), thickness=2):
    """
    Draw a bounding box on the frame using OpenCV.
    """
    cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                  (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                  color, thickness)

def draw_text(frame, text, position, font=cv2.FONT_HERSHEY_DUPLEX, scale=1, color=(255, 0, 0), thickness=3):
    """
    Draw text on the frame using OpenCV.
    """
    cv2.putText(frame, text, position, font, scale, color, thickness)

def calculate_fps(current_time, last_time):
    """
    Calculate FPS based on time difference.
    """
    if last_time == 0:
        return 0
    delta_time = current_time - last_time
    if delta_time <= 0:
        return 0
    return 1 / delta_time


class CentroidTracker:
    """
    Lightweight centroid tracker that assigns temporary IDs to faces.
    Matches bounding boxes between frames based on centroid proximity.
    """

    def __init__(self, max_distance=50, max_disappeared=30):
        """
        Initialize the tracker.
        
        Args:
            max_distance: Maximum centroid distance to match (pixels)
            max_disappeared: Number of frames before deregistering an object
        """
        self.next_object_id = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = {}  # {id: frame count}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.bbox_history = {}  # {id: latest bbox}

    def register(self, centroid, bbox):
        """
        Register a new object with a centroid and bounding box.
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bbox_history[self.next_object_id] = bbox
        next_id = self.next_object_id
        self.next_object_id += 1
        return next_id

    def deregister(self, object_id):
        """
        Remove a tracked object.
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bbox_history[object_id]

    def update(self, bboxes):
        """
        Update the tracker with new bounding boxes.
        Returns dict of {object_id: bbox} for currently tracked objects.
        """

        result = {}

        if len(bboxes) == 0:
            # No detections: increment disappeared for all objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return result

        # Calculate centroids for current bounding boxes
        input_centroids = []
        for bbox in bboxes:
            cx = bbox.origin_x + bbox.width // 2
            cy = bbox.origin_y + bbox.height // 2
            input_centroids.append((cx, cy, bbox))

        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for cx, cy, bbox in input_centroids:
                object_id = self.register((cx, cy), bbox)
                result[object_id] = bbox
            return result

        # Match existing objects to new centroids
        matched_input_indices = set()

        for object_id in list(self.objects.keys()):
            obj_cx, obj_cy = self.objects[object_id]

            # Find nearest input centroid
            min_distance = float('inf')
            nearest_idx = -1

            for idx, (in_cx, in_cy, bbox) in enumerate(input_centroids):
                if idx in matched_input_indices:
                    continue

                dist = math.sqrt((obj_cx - in_cx) ** 2 + (obj_cy - in_cy) ** 2)
                if dist < min_distance:
                    min_distance = dist
                    nearest_idx = idx

            # If found a close match, update the object
            if nearest_idx >= 0 and min_distance < self.max_distance:
                in_cx, in_cy, bbox = input_centroids[nearest_idx]
                self.objects[object_id] = (in_cx, in_cy)
                self.disappeared[object_id] = 0
                self.bbox_history[object_id] = bbox
                result[object_id] = bbox
                matched_input_indices.add(nearest_idx)
            else:
                # Object disappeared
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] <= self.max_disappeared:
                    result[object_id] = self.bbox_history[object_id]
                else:
                    self.deregister(object_id)

        # Register unmatched input centroids as new objects
        for idx, (in_cx, in_cy, bbox) in enumerate(input_centroids):
            if idx not in matched_input_indices:
                object_id = self.register((in_cx, in_cy), bbox)
                result[object_id] = bbox

        return result

    def get_bbox(self, object_id):
        """
        Get the latest bounding box for an object ID.
        """
        return self.bbox_history.get(object_id)


def is_in_collision_zone(bbox, frame_width, zone_percent=0.4):
    """
    Check if a bounding box center is in the collision zone.
    The collision zone is the center portion of the screen.
    
    Args:
        bbox: Bounding box object with origin_x, origin_y, width, height
        frame_width: Total width of the frame (pixels)
        zone_percent: What percentage of frame to consider as center zone (default 40%)
    
    Returns:
        True if bbox center is in the collision zone, False otherwise
    """
    zone_margin = (1 - zone_percent) / 2  # Margin on each side
    zone_left = frame_width * zone_margin
    zone_right = frame_width * (1 - zone_margin)

    center_x = bbox.origin_x + bbox.width / 2
    return zone_left <= center_x <= zone_right


def is_bbox_expanding(bbox_current, bbox_history_list, threshold=1000):
    """
    Checks if bounding box AREA is expanding using a moving average over the last 5 frames.
    Fixes the 'jitter' spam and accounts for vertical approach angles.
    """
    if not bbox_history_list or len(bbox_history_list) < 5:
        return False
    
    # Get average AREA (width * height) of older frames
    older_avg_area = sum((b.width * b.height) for b in bbox_history_list[:-1]) / (len(bbox_history_list) - 1)
    current_area = bbox_current.width * bbox_current.height
    
    # Only return True if the current area is consistently larger than the recent average
    return (current_area - older_avg_area) > threshold


def process_command(command, enrollment_manager, voice_queue, quit_flag, vision_pipeline=None):


    intent = command.get("intent")
    target_name = command.get("target_name")
    global pending_forget_name
    
    try:
        
        # --- WAKE WORD ONLY LOGIC ---
        if intent == INTENT_WAKE_ONLY:
            greetings = [
                "I'm here.", 
                "How can I help you?", 
                "Listening.", 
                "At your service.", 
                "What's up?"
            ]
            voice_queue.speak(random.choice(greetings), voice_queue.PRIORITY_FEEDBACK)
            return

        # --- NEW: CONFIRMATION LOGIC ---
        if pending_forget_name:
            if intent == INTENT_CONFIRM:
                names_stored = get_names_from_PK()
                result = forget_person(pending_forget_name, names_stored)

                # NEW: Trigger the hot-reload
                if vision_pipeline:
                    vision_pipeline.hot_reload()

                voice_queue.speak(result, voice_queue.PRIORITY_FEEDBACK)
                pending_forget_name = None  # Reset state
                return
            elif intent in [INTENT_DENY, INTENT_CANCEL]:
                voice_queue.speak("Deletion cancelled.", voice_queue.PRIORITY_FEEDBACK)
                pending_forget_name = None  # Reset state
                return
            # If they say something else entirely, we just let the command process normally 
            # but we clear the pending state so it doesn't get stuck.
            elif intent != "INTENT_NONE":
                pending_forget_name = None

        # --- 1. CANCEL LOGIC ---
        if intent == INTENT_CANCEL:
            enrollment_manager.cancel()
            # Drain system and info queues, but keep critical warnings
            voice_queue.cancel_queues(voice_queue.PRIORITY_SYSTEM, voice_queue.PRIORITY_INFO)
            voice_queue.speak("Action cancelled.", voice_queue.PRIORITY_FEEDBACK)
            return

        # --- 2. ENROLL LOGIC ---
        if intent == INTENT_ENROLL:
            if not target_name:
                voice_queue.speak("Please say the name to enroll.", voice_queue.PRIORITY_FEEDBACK)
                return
            if enrollment_manager.active:
                voice_queue.speak("Enrollment is already in progress.", voice_queue.PRIORITY_FEEDBACK)
                return
            enrollment_manager.start(target_name, time.time())
            return

        # --- 3. FORGET LOGIC (Fuzzy Match & Lists) ---
        if intent == INTENT_FORGET:
            names_stored = get_names_from_PK()
            if not names_stored:
                voice_queue.speak("Your database is empty.", voice_queue.PRIORITY_FEEDBACK)
                return

            # If user didn't say a name: read the list
            if not target_name:
                name_list = ", ".join([f"Number {i+1}: {name}" for i, name in enumerate(names_stored.keys())])
                voice_queue.speak(f"Who should I forget? I know: {name_list}", voice_queue.PRIORITY_SYSTEM)
                return

            # Fuzzy match (If user says "Ruby", find "Rouby")
            best_matches = difflib.get_close_matches(target_name, names_stored.keys(), n=1, cutoff=0.6)
            
            if best_matches:
                matched_name = best_matches[0]
                pending_forget_name = matched_name
                voice_queue.speak(f"Are you sure you want to forget {matched_name}? Say yes to confirm.", voice_queue.PRIORITY_FEEDBACK)
            else:
                voice_queue.speak(f"I couldn't find {target_name}. Please try again.", voice_queue.PRIORITY_FEEDBACK)
            return

        # --- 4. QUIT LOGIC ---
        if intent == INTENT_QUIT:
            quit_flag[0] = True
            return

    finally:
        # ALWAYS resume the voice queue after processing a command
        if hasattr(voice_queue, 'resume'):
            voice_queue.resume()


def handle_manual_input(intent_type, command_queue):
    """Handle manual terminal input in a daemon thread."""
    try:
        name = input("Enter name: ").strip()
        if name:
            command = {"intent": intent_type, "target_name": name}
            command_queue.put(command)
    except Exception as e:
        logger.exception(f"Manual input error: {e}")