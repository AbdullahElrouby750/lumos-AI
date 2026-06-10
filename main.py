import cv2
import time

import logging

from src.core.nova_audio import get_voice_queue
from src.core.nova_logger import setup_logger
from src.core.nova_vision_pipeline import VisionPipeline
from src.core.utils import draw_text, calculate_fps
from src.network.nova_server import LumosServer
from src.workers.nova_ai_worker import NovaAIWorker
from src.workers.nova_enrollment_worker import NovaEnrollmentWorker
from src.workers.nova_yolo_worker import NovaYoloWorker


logger = logging.getLogger(__name__)

def main():
    """Main CEO script for Lumos with server-driven workers and a clean edge pipeline."""
    setup_logger()
    server = LumosServer()
    server.start()

    yolo_worker = NovaYoloWorker(server)
    ai_worker = NovaAIWorker(server)
    enrollment_worker = NovaEnrollmentWorker(server)

    voice_queue = get_voice_queue()
    voice_queue.set_server(server)

    vision_pipeline = VisionPipeline(
        voice_queue,
        server=server,
        yolo_worker=yolo_worker,
        ai_worker=ai_worker,
        enrollment_worker=enrollment_worker,
    )
    server.set_command_callback(vision_pipeline.on_command_received)
    if vision_pipeline.detector.detector is None or vision_pipeline.recognizer.facenet_model is None:
        logger.error("Failed to initialize detector or recognizer. Exiting.")
        voice_queue.speak("Initialization failed", voice_queue.PRIORITY_DANGER)
        vision_pipeline.close()
        voice_queue.stop()
        server.stop()
        yolo_worker.stop()
        ai_worker.stop()
        return

    try:
        cap = cv2.VideoCapture("http://10.42.0.238:8888/stream.mjpg")        
        # --- BUG A-3 FIX (Part 1): Hardware Request ---
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise Exception("Camera not accessible")
    except Exception as e:
        logger.error(f"Error opening camera: {e}")
        voice_queue.speak("Camera error", voice_queue.PRIORITY_DANGER)
        vision_pipeline.close()
        voice_queue.stop()
        server.stop()
        yolo_worker.stop()
        ai_worker.stop()
        return

    last_time = 0
    quit_flag = [False]

    # --- BUG A-3 FIX (Part 2): Throttle Setup ---
    TARGET_FPS = 30
    FRAME_TIME_TARGET = 1.0 / TARGET_FPS
    # --------------------------------------------
    
    logger.info("Lumos: Ready for edge commands from the server.")

    try:
        while cap.isOpened() and not quit_flag[0]:
            current_time = time.time()
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            frame = vision_pipeline.process_frame(frame, current_time)
            
            # --- BUG F-1 FIX (Part 3): Listen to the CEO ---
            if vision_pipeline.quit_requested:
                logger.info("Shutdown signal received from pipeline. Breaking loop.")
                break
            # -----------------------------------------------

            fps = calculate_fps(current_time, last_time)
            last_time = current_time
            draw_text(frame, f"FPS: {fps}", (20, 50), scale=1.0, color=(255, 0, 0))

            cv2.imshow("Lumos Face Detection - Spam Filter Active", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # --- BUG A-3 FIX (Part 3): Software Throttle Execution ---
            # Calculate how long this frame took. If it was faster than 33ms, sleep the difference.
            process_time = time.time() - current_time 
            sleep_time = FRAME_TIME_TARGET - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            # ---------------------------------------------------------

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        logger.info("Lumos shutting down")
        cap.release()
        cv2.destroyAllWindows()
        vision_pipeline.close()
        server.stop()
        yolo_worker.stop()
        ai_worker.stop()
        enrollment_worker.stop()
        time.sleep(2)
        voice_queue.stop()


if __name__ == "__main__":
    main()
