"""
On-demand AI worker for Lumos.

This worker listens for explicit user intents, reads the latest scene image from
RAM disk, and delegates processing to the brain module or OCR module without
blocking the main camera loop.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict

from brain_module import describe_scene
from OCR import get_text_from_image
from nova_audio import VoiceQueue
from nova_network_models import BaseEvent, OCRResultEvent
from nova_server import LumosServer

logger = logging.getLogger(__name__)

RAM_DISK_SCENE_PATH = Path("/dev/shm/latest_scene.jpg")
FALLBACK_SCENE_PATH = Path("temp_scene.jpg")

INTENT_SCENE = "INTENT_SCENE"
INTENT_TEXT = "INTENT_TEXT"


class NovaAIWorker:
    def __init__(self, server: LumosServer, queue_size: int = 8, command_timeout: float = 1.0):
        self.server = server
        self._command_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._command_timeout = command_timeout
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="NovaAIWorker")
        self._worker_thread.start()
        logger.info("NovaAIWorker started with queue size=%d", queue_size)

    def enqueue_command(self, command: Dict[str, Any]) -> bool:
        try:
            self._command_queue.put_nowait(command)
            logger.debug("Enqueued AI command: %s", command)
            return True
        except queue.Full:
            logger.warning("AI worker queue is full. Dropping command: %s", command)
            return False

    def stop(self) -> None:
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)
        logger.info("NovaAIWorker stopped")

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                command = self._command_queue.get(timeout=self._command_timeout)
            except queue.Empty:
                continue

            try:
                self._process_command(command)
            except Exception as exc:
                logger.exception("Unexpected error processing AI command")
                self._send_warning("Network connection lost. Cannot process the request right now.")
            finally:
                self._command_queue.task_done()

    def _process_command(self, command: Dict[str, Any]) -> None:
        intent = command.get("intent")

        if intent == INTENT_SCENE:
            self._handle_scene_request(command)
        elif intent == INTENT_TEXT:
            self._handle_text_request(command)
        else:
            logger.warning("Unsupported AI intent received: %s", intent)

    def _handle_scene_request(self, command: Dict[str, Any]) -> None:
        image_path = self._resolve_scene_path()
        if image_path is None:
            self._send_warning("Network connection lost. Cannot process the request right now.")
            return

        try:
            description = describe_scene()
            event = BaseEvent.create(
                "SCENE_DESCRIPTION",
                {"text": description, "intent": INTENT_SCENE},
                priority=VoiceQueue.PRIORITY_LOW,
            )
            self.server.send_event(event)
            logger.info("Scene description event sent")
        except Exception as exc:
            logger.exception("Scene description failed")
            self._send_warning("Network connection lost. Cannot process the request right now.")

    def _handle_text_request(self, command: Dict[str, Any]) -> None:
        image_path = self._resolve_scene_path()
        if image_path is None:
            self._send_warning("Network connection lost. Cannot process the request right now.")
            return

        try:
            ocr_text = get_text_from_image(str(image_path))
            event = OCRResultEvent.create(
                "OCR_RESULT",
                {"text": ocr_text, "intent": INTENT_TEXT},
                priority=VoiceQueue.PRIORITY_INFO,
            )
            self.server.send_event(event)
            logger.info("OCR result event sent")
        except Exception as exc:
            logger.exception("Text recognition failed")
            self._send_warning("Network connection lost. Cannot process the request right now.")

    def _resolve_scene_path(self) -> Path | None:
        if RAM_DISK_SCENE_PATH.exists():
            return RAM_DISK_SCENE_PATH

        if FALLBACK_SCENE_PATH.exists():
            logger.warning("RAM disk scene not found at %s. Falling back to %s", RAM_DISK_SCENE_PATH, FALLBACK_SCENE_PATH)
            return FALLBACK_SCENE_PATH

        logger.warning("No scene image available at %s or fallback %s", RAM_DISK_SCENE_PATH, FALLBACK_SCENE_PATH)
        return None

    def _send_warning(self, message: str) -> None:
        event = BaseEvent.create(
            "AI_WORKER_ERROR",
            {"text": message},
            priority=VoiceQueue.PRIORITY_WARNING,
        )
        self.server.send_event(event)
