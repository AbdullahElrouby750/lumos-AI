"""
Thread-safe YOLO worker for Lumos hazard detection.

This worker runs in a dedicated daemon thread, consumes frames from an input
queue, throttles inference to a maximum frame rate, and sends structured
SocialAlertEvent messages to the FastAPI server.
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.core.nova_audio import VoiceQueue
from config.nova_config_manager import HazardConfig, NovaConfigManager
from src.network.nova_network_models import SocialAlertEvent
from src.network.nova_server import LumosServer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/yolo11n.pt")


class NovaYoloWorker:
    def __init__(self, server: LumosServer, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.server = server
        self._config_manager = NovaConfigManager.get_instance()
        self._model_path = Path(model_path)
        self._frame_queue: queue.Queue[tuple[float, Any]] = queue.Queue(maxsize=self._config_manager.get_config().queue_size)
        self._inference_semaphore = threading.BoundedSemaphore(1)
        self._stop_event = threading.Event()
        self._last_inference_time = 0.0
        self._last_alert_time = 0.0
        self._last_safety_alert = ""
        self._model = self._load_model()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="NovaYoloWorker")
        self._worker_thread.start()

    def _load_model(self) -> YOLO | None:
        if not self._model_path.exists():
            logger.warning("YOLO model file not found at %s. YOLO worker will remain disabled.", self._model_path)
            return None

        try:
            model = YOLO(str(self._model_path))
            logger.info("Loaded YOLO model from %s", self._model_path)
            return model
        except Exception as exc:
            logger.warning("Failed to load YOLO model from %s: %s", self._model_path, exc)
            return None

    def enqueue_frame(self, frame: Any) -> bool:
        """Add a new frame to the YOLO input queue."""
        try:
            self._frame_queue.put_nowait((time.time(), frame))
            return True
        except queue.Full:
            logger.warning("YOLO worker queue is full. Dropping frame to preserve throughput.")
            return False

    def stop(self) -> None:
        """Gracefully stop the YOLO worker thread."""
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)

    def _worker_loop(self) -> None:
        config = self._config_manager.get_config()
        cooldown = 1.0 / max(config.max_fps, 1.0)

        while not self._stop_event.is_set():
            try:
                _, frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._model is None:
                self._frame_queue.task_done()
                continue

            now = time.time()
            if now - self._last_inference_time < cooldown:
                self._frame_queue.task_done()
                continue

            if not self._inference_semaphore.acquire(blocking=False):
                self._frame_queue.task_done()
                continue

            try:
                self._run_inference(frame, config, now)
            finally:
                self._inference_semaphore.release()
                self._last_inference_time = time.time()
                self._frame_queue.task_done()

    def _run_inference(
        self,
        frame: Any,
        config: HazardConfig,
        now: float,
    ) -> None:
        try:
            results = list(self._model.predict(frame, conf=config.detection_confidence, stream=True, verbose=False))
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            return

        self._process_results(frame, results, config, now)

    def _process_results(self, frame: Any, results: list[Any], config: HazardConfig, now: float) -> None:
        if frame is None:
            return

        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            logger.warning("Invalid frame shape for YOLO worker.")
            return

        width = float(shape[1])
        current_emergency = False
        person_count_center = 0

        for result in results:
            for box in getattr(result, "boxes", []):
                label = self._get_label(box)
                x1, y1, x2, y2 = self._get_box_coordinates(box)
                distance = self._estimate_distance(y1, y2)
                center_x = (x1 + x2) / 2.0
                in_path = width / 3.0 < center_x < 2.0 * width / 3.0

                if label == "person" and in_path:
                    person_count_center += 1

                if label in config.danger_objects and distance < config.danger_distance:
                    current_emergency = True
                    payload = {
                        "label": label,
                        "distance": round(distance, 1),
                        "message": f"Danger: {label} very close.",
                    }
                    self._send_event(
                        "OBJECT_DETECTION",
                        payload,
                        VoiceQueue.PRIORITY_CRITICAL,
                    )
                    break

                if label in config.trip_hazards and distance < config.hazard_distance and in_path:
                    if label != self._last_safety_alert or now - self._last_alert_time > config.hazard_cooldown:
                        payload = {
                            "label": label,
                            "distance": round(distance, 1),
                            "message": f"Caution: {label} ahead.",
                        }
                        self._send_event(
                            "OBJECT_DETECTION",
                            payload,
                            VoiceQueue.PRIORITY_WARNING,
                        )
                        self._last_safety_alert = label
                        self._last_alert_time = now

            if current_emergency:
                break

        if not current_emergency and person_count_center >= config.crowd_threshold:
            if now - self._last_alert_time > config.hazard_cooldown * 2.0:
                payload = {
                    "label": "crowd",
                    "count": person_count_center,
                    "message": "Crowded area ahead.",
                }
                self._send_event(
                    "CROWD_ALERT",
                    payload,
                    VoiceQueue.PRIORITY_INFO,
                )
                self._last_alert_time = now

    def _get_label(self, box: Any) -> str:
        label_index = None
        if getattr(box, "cls", None) is not None:
            label_index = int(box.cls[0])

        if self._model is not None and label_index is not None:
            try:
                return str(self._model.names[label_index])
            except Exception:
                pass

        return "unknown"

    def _get_box_coordinates(self, box: Any) -> tuple[float, float, float, float]:
        xyxy = getattr(box, "xyxy", None)
        if xyxy is None:
            return 0.0, 0.0, 0.0, 0.0

        values = list(xyxy[0])
        return tuple(float(v) for v in values)

    def _estimate_distance(self, box_y1: float, box_y2: float) -> float:
        pixel_height = max(box_y2 - box_y1, 1.0)
        focal_length = 160.0
        real_height = 170.0
        return (real_height * focal_length) / (pixel_height * 100.0)

    def _send_event(self, event_type: str, payload: dict[str, Any], priority: int = VoiceQueue.PRIORITY_INFO) -> None:
        event = SocialAlertEvent.create(event_type, payload, priority=priority)
        self.server.send_event(event)
