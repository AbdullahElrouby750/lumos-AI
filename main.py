import cv2
import threading
import time
from queue import Empty, Queue

from nova_audio import VoiceQueue
from nova_listener import CommandListener
from nova_enrollment import EnrollmentManager
from nova_vision_pipeline import VisionPipeline
from nova_commands import INTENT_ENROLL, INTENT_FORGET, INTENT_QUIT
from utils import calculate_fps, draw_text, process_command, handle_manual_input

COMMAND_HOST = "127.0.0.1"
COMMAND_PORT = 65432


def main():
    """Main CEO script for Lumos with voice command input and guided enrollment."""
    voice_queue = VoiceQueue()
    voice_queue.speak("Lumos starting", VoiceQueue.PRIORITY_INFO)

    vision_pipeline = VisionPipeline(voice_queue)
    if vision_pipeline.detector.detector is None or vision_pipeline.recognizer.facenet_model is None:
        print("Failed to initialize detector or recognizer. Exiting.")
        voice_queue.speak("Initialization failed", VoiceQueue.PRIORITY_DANGER)
        vision_pipeline.close()
        voice_queue.stop()
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Camera not accessible")
    except Exception as e:
        print(f"Error opening camera: {e}")
        voice_queue.speak("Camera error", VoiceQueue.PRIORITY_DANGER)
        vision_pipeline.close()
        voice_queue.stop()
        return

    command_queue = Queue()
    command_listener = CommandListener(command_queue, voice_queue)
    command_listener.start()

    enrollment_manager = EnrollmentManager(voice_queue)

    last_time = 0
    quit_flag = [False]

    print("Lumos: Ready for voice commands. Say 'enroll' or 'forget'.")
    print(f"Listening for commands on {COMMAND_HOST}:{COMMAND_PORT}")
    print("Press 's' for manual enroll, 'f' for manual forget, 'q' to quit.")

    try:
        while cap.isOpened() and not quit_flag[0]:
            current_time = time.time()
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            enrollment_manager.update(frame, current_time)

            try:
                while True:
                    command = command_queue.get_nowait()
                    process_command(command, enrollment_manager, voice_queue, quit_flag)
            except Empty:
                pass

            frame = vision_pipeline.process_frame(frame, current_time)

            fps = calculate_fps(current_time, last_time)
            last_time = current_time
            draw_text(frame, f"FPS: {int(fps)}", (20, 50), scale=1.0, color=(255, 0, 0))

            cv2.imshow("Lumos Face Detection - Spam Filter Active", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Manual enroll
                thread = threading.Thread(target=handle_manual_input, args=(INTENT_ENROLL, command_queue), daemon=True)
                thread.start()
            elif key == ord('f'):
                # Manual forget
                thread = threading.Thread(target=handle_manual_input, args=(INTENT_FORGET, command_queue), daemon=True)
                thread.start()

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        command_listener.stop()
        cap.release()
        cv2.destroyAllWindows()
        vision_pipeline.close()

        voice_queue.speak("Lumos shutting down", VoiceQueue.PRIORITY_INFO)
        while voice_queue.get_queue_size() > 0:
            time.sleep(0.5)
        time.sleep(2)
        voice_queue.stop()


if __name__ == "__main__":
    main()
