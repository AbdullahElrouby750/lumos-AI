import cv2
import time

# Constants (preserved from existing code)
KNOWN_WIDTH = 14.0  # cm
FOCAL_LENGTH = 678.57  # Preserved calibrated value

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