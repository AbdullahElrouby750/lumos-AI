import cv2
import time

from src.core.nova_audio import get_voice_queue
from src.core.nova_vision_pipeline import VisionPipeline
from src.core.utils import draw_text, calculate_fps
from src.modules.nova_enrollment import EnrollmentManager
from src.network.nova_server import LumosServer
from src.workers.nova_ai_worker import NovaAIWorker
from src.workers.nova_yolo_worker import NovaYoloWorker


def main():
    """Main CEO script for Lumos with server-driven workers and a clean edge pipeline."""
    server = LumosServer()
    server.start()

    yolo_worker = NovaYoloWorker(server)
    ai_worker = NovaAIWorker(server)

    voice_queue = get_voice_queue()
    voice_queue.set_server(server)

    vision_pipeline = VisionPipeline(
        voice_queue,
        server=server,
        yolo_worker=yolo_worker,
        ai_worker=ai_worker,
    )
    server.set_command_callback(vision_pipeline.on_command_received)
    if vision_pipeline.detector.detector is None or vision_pipeline.recognizer.facenet_model is None:
        print("Failed to initialize detector or recognizer. Exiting.")
        voice_queue.speak("Initialization failed", voice_queue.PRIORITY_DANGER)
        vision_pipeline.close()
        voice_queue.stop()
        server.stop()
        yolo_worker.stop()
        ai_worker.stop()
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Camera not accessible")
    except Exception as e:
        print(f"Error opening camera: {e}")
        voice_queue.speak("Camera error", voice_queue.PRIORITY_DANGER)
        vision_pipeline.close()
        voice_queue.stop()
        server.stop()
        yolo_worker.stop()
        ai_worker.stop()
        return

    enrollment_manager = EnrollmentManager(voice_queue, vision_pipeline=vision_pipeline)

    last_time = 0
    quit_flag = [False]

    print("Lumos: Ready for edge commands from the server.")

    try:
        while cap.isOpened() and not quit_flag[0]:
            current_time = time.time()
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            enrollment_manager.update(frame, current_time)

            frame = vision_pipeline.process_frame(frame, current_time)

            fps = calculate_fps(current_time, last_time)
            last_time = current_time
            draw_text(frame, f"FPS: {fps}", (20, 50), scale=1.0, color=(255, 0, 0))

            cv2.imshow("Lumos Face Detection - Spam Filter Active", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vision_pipeline.close()
        server.stop()
        yolo_worker.stop()
        ai_worker.stop()
        voice_queue.speak("Lumos shutting down", voice_queue.PRIORITY_FEEDBACK)
        time.sleep(2)
        voice_queue.stop()


if __name__ == "__main__":
    main()
