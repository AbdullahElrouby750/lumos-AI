import cv2
from ultralytics import YOLO
import pyttsx3 as tts
import threading
import time

def speak(text):
    engine = tts.init()
    engine.setProperty('rate', 190)
    engine.say(text)
    engine.runAndWait()

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# The "Memory" variable: stores (label, position)
last_state = (None, None)

print("Lumos HD Voice System Active...")
last_time = 0

while cap.isOpened():
    current_time = time.time()
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    # We only care about the highest confidence object for this test
    results = model(frame, stream=True, conf=0.45)

    width = frame.shape[1]
    
    for r in results:
        # Sort boxes by confidence so we deal with the most certain object first
        boxes = sorted(r.boxes, key=lambda x: x.conf, reverse=True)
        
        if len(boxes) > 0:
            box = boxes[0] # Focus on the most prominent object
            label = model.names[int(box.cls[0])]
            
            # Determine Position
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2
            
            if center_x < (width / 3): current_pos = "LEFT"
            elif center_x > (2 * width / 3): current_pos = "RIGHT"
            else: current_pos = "CENTER"

            # STATE CHANGE CHECK:
            # If the label OR the position is different from last time, report it.
            current_state = (label, current_pos)
            
            if current_state != last_state:
                
                announcement = f"{label} detected {current_pos}"
                print(f"Speaking: {announcement}")
                
                # Visuals
                cv2.rectangle(frame, (int(x1), int(box.xyxy[0][1])), (int(x2), int(box.xyxy[0][3])), (255, 0, 0), 3)
                
                threading.Thread(target=speak, args=(announcement,), daemon=True).start()
                
                last_state = current_state # Update the memory
            cv2.rectangle(frame, (int(x1), int(box.xyxy[0][1])), (int(x2), int(box.xyxy[0][3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {current_pos}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            fps = 1 / (current_time - last_time)
            last_time = current_time
            cv2.putText(frame, f"FPS: {int(fps)}",(20,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),3)


    cv2.imshow("Lumos Voice Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()