import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from deepface import DeepFace
import numpy as np
import threading
import pyttsx3 as tts

# 1. Initialize Voice and Load Brain
def speak(text):
    engine = tts.init()
    engine.say(text)
    engine.runAndWait()

with open("nova_brain.pkl", "rb") as f:
    known_faces = pickle.load(f)

print(f"Loaded {len(known_faces)} known faces from the brain.")  # Debug: show how many faces are loaded

# 2. Setup MediaPipe Face Detector (Your "Gatekeeper")
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# OPTIMIZATION: Load Facenet512 model ONCE before loop
# Loading model inside loop = reloading it every frame = SLOW
# Pre-loading = reuse the same model, ~10x faster
facenet_model = DeepFace.build_model("Facenet512")

cap = cv2.VideoCapture(0)
last_recognized = ""

KNOWN_WIDTH = 14.0 # cm
FOCAL_LENGTH = 678.57 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)

    # Convert for MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            
            # --- START RECOGNITION LOGIC ---
            # We only run recognition if a face is actually there
            try:
                # 1. Prepare the crop for the model (Facenet512 needs 160x160)
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                face_crop = frame[y:y+h, x:x+w]
                
                # --- DISTANCE CALCULATION ---
                # Using the width 'w' from your bounding box
                distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / w
                distance_m = distance_cm / 100.0
                
                print(f"width for calculating the focal length: {w}") # Debug: show the width of the detected face
                
                face_resize = cv2.resize(face_crop, (160, 160))
                face_array = np.expand_dims(face_resize, axis=0)
                face_array = face_array / 255.0 # Normalize pixels

                # 2. Get the embedding directly from the pre-loaded model
                # This avoids the DeepFace.represent argument errors entirely!
                # Notice the extra '.model' added here!
                current_embedding = facenet_model.model.predict(face_array, verbose=0)[0]
                best_match = "Unknown"
                min_dist = 0.6 

                for name, embeddings in known_faces.items():
                    for db_emb in embeddings:
                        # Manually calculate Cosine Distance (Math-based, extremely fast)
                        a = np.matmul(current_embedding, db_emb)
                        b = np.sum(np.multiply(current_embedding, current_embedding))
                        c = np.sum(np.multiply(db_emb, db_emb))
                        dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

                        if dist < min_dist:
                            min_dist = dist
                            best_match = name
                # Voice Alert on Change
                if best_match != last_recognized and best_match != "Unknown":
                    threading.Thread(target=speak, args=(f"I see {best_match} at a distance of {distance_m:.2f} meters.",), daemon=True).start()
                    last_recognized = best_match

                # Visual Feedback
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                              (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), color, 2)
                cv2.putText(frame, f"Recognized: {best_match}", (bbox.origin_x, bbox.origin_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print(f"Recognition error: {e}") # Debug: show any errors during recognition
                pass # Recognition failed this frame, ignore and move on

    cv2.imshow("Lumos Live Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()