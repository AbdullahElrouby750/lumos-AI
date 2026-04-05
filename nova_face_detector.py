import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyttsx3 as tts
import threading
import time
import os

from nove_face_encoder import build_nova_brain as face_encoder
from nove_forget import forget_person, get_names_from_PK


DB_PATH = "face_db"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

def speak(text):
    engine = tts.init()
    engine.setProperty('rate', 190)
    engine.say(text)
    engine.runAndWait()

# 1. Setup the Options
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

last_seen = False
last_time = 0

print("Lumos: Press 's' to enroll a new person, 'f' to forget a person, 'q' to quit.")

is_enrolling = False
enroll_name = ""
frames_captured = 0
last_capture_time = 0

while cap.isOpened():
    current_time = time.time()
    success, frame = cap.read()
    if not success: break
    
    # MediaPipe Tasks require a special 'Image' object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 2. Detect
    detection_result = detector.detect(mp_image)
    
    current_seen = False
    # 3. Process Results
    if detection_result.detections:
        current_seen = True
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            # Draw on the ORIGINAL BGR frame for OpenCV
            cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 255), 2)

    if current_seen and not last_seen:
        threading.Thread(target=speak, args=("face detected",),daemon=True).start()
    
    fps = 1 / (current_time - last_time)
    last_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}",(20,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),3)
    
    last_seen = current_seen
    cv2.imshow("Lumos Modern Face Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # --- ENROLLMENT LOGIC START ---
    if key == ord('s') and not is_enrolling:
        enroll_name = input("Enter name: ")
        is_enrolling = True
        frames_captured = 0
        threading.Thread(target=speak, args=(f"Recording {enroll_name}",), daemon=True).start()

    # --- THE NON-BLOCKING CAPTURE LOGIC ---
    if is_enrolling:
        # Only capture if 1 second has passed AND we haven't hit 5 frames
        if time.time() - last_capture_time > 3.0:
            file_name = f"{DB_PATH}/{enroll_name}_{frames_captured}.jpg"
            cv2.imwrite(file_name, frame) # Use the current 'frame' from the main loop
            
            frames_captured += 1
            last_capture_time = time.time()
            
            # Speak frame count without blocking
            threading.Thread(target=speak, args=(f"Frame {frames_captured}",), daemon=True).start()
            print(f"Captured {frames_captured}/5")

        # Visual feedback on the live stream
        cv2.putText(frame, f"ENROLLING: {frames_captured}/5", (20, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

        # Check if we are finished
        if frames_captured >= 5:
            is_enrolling = False
            threading.Thread(target=speak, args=("Enrollment complete",), daemon=True).start()
            threading.Thread(target=face_encoder, args=(enroll_name,), daemon=True).start()
            
    # --- ENROLLMENT LOGIC END ---
    
    # --- FORGET LOGIC START ---
    if key == ord('f'):
        names_stored = get_names_from_PK()
        names = list(names_stored.keys()) if names_stored else []
        if names:
            print(f"Current people stored in memory: {names}")
            target_name = input("Enter the EXACT name of the person to forget: ")
            result = forget_person(target_name, names_stored)
        
        threading.Thread(target=speak, args=(result), daemon=True).start()

    if key == ord('q'): break

detector.close()
cap.release()