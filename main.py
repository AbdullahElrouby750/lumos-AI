import cv2
import mediapipe as mp
import time
import threading
from nova_face_detector import FaceDetector
from nove_face_rec import FaceRecognizer
from nova_audio import VoiceQueue
from utils import (
    draw_bounding_box, draw_text, calculate_fps,
    CentroidTracker, is_in_collision_zone, is_bbox_expanding
)

def main():
    """
    Main CEO script for Lumos with spam filter and collision zone detection.
    Initializes camera, runs detection loop with centroid tracking,
    async recognition, and intelligent voice alerts.
    Ensures main loop stays above 30 FPS.
    """
    # Initialize VoiceQueue
    voice_queue = VoiceQueue()
    voice_queue.speak("Lumos starting", VoiceQueue.PRIORITY_INFO)

    # Initialize FaceDetector and FaceRecognizer
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if detector.detector is None or recognizer.facenet_model is None:
        print("Failed to initialize detector or recognizer. Exiting.")
        voice_queue.speak("Initialization failed", VoiceQueue.PRIORITY_DANGER)
        detector.close()
        recognizer.close()
        voice_queue.stop()
        return

    # Initialize camera with graceful failure
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

    name_cooldowns = {} # Dictionary to remember when we last said a name
    COOLDOWN_TIME = 30.0 # How many seconds to wait before saying a name again

    # Initialize CentroidTracker
    tracker = CentroidTracker(max_distance=200, max_disappeared=60)

    # State management
    last_time = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    recognition_threads = {}  # {face_id: thread}
    recognition_results = {}  # {face_id: name}
    recognized_ids = set()    # Track which IDs have been voice-announced
    previous_bboxes = {}      # {face_id: previous bbox} for expansion detection
    alerted_approaching_ids = set()  # Track which IDs already got "approaching" alert

    print("Lumos: Starting face detection with spam filter. Press 'q' to quit.")
    print(f"Frame dimensions: {frame_width}x{frame_height}")
    print(f"Collision zone: center 40% ({int(frame_width*0.3)}-{int(frame_width*0.7)})")

    try:
        while cap.isOpened():
            current_time = time.time()
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            # Detect faces
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(mp_image)

            # Extract bounding boxes from detections
            detected_bboxes = []
            if detection_result and detection_result.detections:
                detected_bboxes = [d.bounding_box for d in detection_result.detections]

            # Update tracker with current detections
            tracked_ids_with_bboxes = tracker.update(detected_bboxes)

            # Process each tracked face
            for face_id, bbox in tracked_ids_with_bboxes.items():
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                # Ensure crop is within frame bounds
                y_start = max(0, y)
                y_end = min(frame.shape[0], y + h)
                x_start = max(0, x)
                x_end = min(frame.shape[1], x + w)

                if y_end > y_start and x_end > x_start:
                    face_crop = frame[y_start:y_end, x_start:x_end].copy()

                    # If this is a NEW face, spawn recognition thread
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

                # Check if recognition has finished for this ID
                if face_id in recognition_results:
                    name = recognition_results[face_id]

                    # ===== ALERT LOGIC =====
                    # CONDITION 1: Newly recognized face (with Name-Level Cooldown)
                    if name != "Unknown":
                        last_spoken_time = name_cooldowns.get(name, 0)
                        
                        # Only speak if we haven't said this name in the last 30 seconds
                        if current_time - last_spoken_time > COOLDOWN_TIME:
                            voice_queue.speak(
                                f"I see {name}",
                                VoiceQueue.PRIORITY_SOCIAL
                            )
                            name_cooldowns[name] = current_time # Update the memory
                            
                        recognized_ids.add(face_id)
                        alerted_approaching_ids.discard(face_id)

                    # CONDITION 2: Recognized face in collision zone and expanding (approaching)
                    if name != "Unknown" and face_id in recognized_ids:
                        if is_in_collision_zone(bbox, frame_width, zone_percent=0.4):
                            prev_bbox = previous_bboxes.get(face_id)
                            if is_bbox_expanding(bbox, prev_bbox, threshold=10):
                                # Only alert once per approach cycle
                                if face_id not in alerted_approaching_ids:
                                    voice_queue.speak(
                                        f"{name} approaching",
                                        VoiceQueue.PRIORITY_DANGER
                                    )
                                    alerted_approaching_ids.add(face_id)
                        else:
                            # Left collision zone, reset alert
                            alerted_approaching_ids.discard(face_id)

                    # Draw visualization
                    if name != "Unknown":
                        color = (0, 255, 0)  # Green for recognized
                        label = f"ID:{face_id} {name}"
                    else:
                        color = (255, 255, 0)  # Yellow for recognizing
                        label = f"ID:{face_id} (recognizing...)"

                    draw_bounding_box(frame, bbox, color=color, thickness=2)
                    draw_text(frame, label, (bbox.origin_x, bbox.origin_y - 10), scale=0.8, color=color)

                # Update previous bbox for next frame
                previous_bboxes[face_id] = bbox

            # Clean up state for faces that left
            current_tracked_ids = set(tracked_ids_with_bboxes.keys())
            for fid in list(recognized_ids):
                if fid not in current_tracked_ids:
                    recognized_ids.discard(fid)
                    previous_bboxes.pop(fid, None)
                    recognition_results.pop(fid, None)
                    recognition_threads.pop(fid, None)
                    alerted_approaching_ids.discard(fid)

            # Draw collision zone (debug visualization: grey vertical lines)
            zone_left = int(frame_width * 0.3)
            zone_right = int(frame_width * 0.7)
            cv2.line(frame, (zone_left, 0), (zone_left, frame_height), (100, 100, 100), 1)
            cv2.line(frame, (zone_right, 0), (zone_right, frame_height), (100, 100, 100), 1)

            # Calculate and display FPS
            fps = calculate_fps(current_time, last_time)
            last_time = current_time
            draw_text(frame, f"FPS: {int(fps)}", (20, 50), scale=1.0, color=(255, 0, 0))

            # Show frame
            cv2.imshow("Lumos Face Detection - Spam Filter Active", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup hardware
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        recognizer.close()

        # Send the final message
        voice_queue.speak("Lumos shutting down", VoiceQueue.PRIORITY_INFO)

        # SMART WAIT: Don't stop the thread if there are still messages waiting in line!
        while voice_queue.get_queue_size() > 0:
            time.sleep(0.5)

        time.sleep(2)  # Give the TTS engine 2 seconds to physically speak the final words
        voice_queue.stop()

if __name__ == "__main__":
    main()