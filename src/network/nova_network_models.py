"""
Lumos Nervous System - Network Models

Pydantic models for event-based JSON messages used in the Lumos WebSocket and REST API.
These models define the structured communication protocol for the asynchronous FastAPI server.
"""

from typing import Any, Dict, Union
from pydantic import BaseModel
import time

from src.core.nova_audio import VoiceQueue


class BaseEvent(BaseModel):
    """
    Base event model for all Lumos network messages.

    This provides the core structure for event multiplexing over WebSocket connections,
    ensuring type safety and consistent timestamping across the nervous system.
    """
    type: str
    payload: Dict[str, Any]
    timestamp: float
    priority: int = VoiceQueue.PRIORITY_INFO

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: Dict[str, Any],
        priority: int = VoiceQueue.PRIORITY_INFO,
    ) -> 'BaseEvent':
        """Factory method to create an event with current timestamp and voice priority."""
        return cls(
            type=event_type,
            payload=payload,
            timestamp=time.time(),
            priority=priority,
        )


class SocialAlertEvent(BaseEvent):
    """Event for social interaction alerts (e.g., face recognition results)."""
    type: str = "SOCIAL_ALERT"


class SystemCommandEvent(BaseEvent):
    """Event for system-level commands and status updates."""
    type: str = "SYSTEM_COMMAND"


class OCRResultEvent(BaseEvent):
    """Event for OCR (Optical Character Recognition) results."""
    type: str = "OCR_RESULT"


# Type alias for any event type
Event = Union[SocialAlertEvent, SystemCommandEvent, OCRResultEvent, BaseEvent]