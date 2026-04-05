import cv2
import mediapipe as mp
import time
from nova_face_detector import FaceDetector
from utils import draw_bounding_box, draw_text, calculate_fps

def main():
    """
    Main CEO script for Lumos.
    Initializes camera, runs detection loop, orchestrates modules.
    Ensures main loop stays above 30 FPS.
    """
    # Initialize FaceDetector
    detector = FaceDetector()
    if detector.detector is None:
        print("Failed to initialize FaceDetector. Exiting.")
        return

    # Initialize camera with graceful failure
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Camera not accessible")
    except Exception as e:
        print(f"Error opening camera: {e}")
        detector.close()
        return

    last_time = 0
    print("Lumos: Starting face detection. Press 'q' to quit.")

    try:
        while cap.isOpened():
            current_time = time.time()
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            # Convert to MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Detect faces
            detection_result = detector.detect(mp_image)

            # Process detections: draw bounding boxes
            if detection_result and detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    draw_bounding_box(frame, bbox)

            # Calculate and display FPS
            fps = calculate_fps(current_time, last_time)
            last_time = current_time
            draw_text(frame, f"FPS: {int(fps)}", (20, 50))

            # Show frame
            cv2.imshow("Lumos Face Detection", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

if __name__ == "__main__":
    main()