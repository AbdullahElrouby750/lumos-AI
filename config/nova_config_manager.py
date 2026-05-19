"""
Nova configuration manager for the Lumos YOLO hazard subsystem.

This module implements a read-once RAM cache for the hazard configuration file
so high-frequency vision loops avoid repeated SD card or disk access.
"""

import json
import logging
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# This guarantees it always finds the JSON right next to the python script
CONFIG_PATH = Path(__file__).parent / "hazards_config.json"


class HazardConfig(BaseModel):
    danger_objects: list[str] = Field(default_factory=lambda: [
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "fire hydrant",
    ])
    trip_hazards: list[str] = Field(default_factory=lambda: [
        "chair",
        "bench",
        "potted plant",
        "suitcase",
        "backpack",
        "box",
        "person",
        "dog",
        "cat",
        "stairs",
        "step",
        "shopping cart",
        "stroller",
    ])
    danger_distance: float = 4.0
    hazard_distance: float = 1.8
    hazard_cooldown: float = 4.0
    crowd_threshold: int = 5
    detection_confidence: float = 0.35
    max_fps: float = 2.0
    queue_size: int = 4


class NovaConfigManager:
    """Singleton manager for cached hazard configuration."""

    _instance: ClassVar[Optional["NovaConfigManager"]] = None

    def __init__(self, config_path: Path = CONFIG_PATH):
        self._config_path = config_path
        self._config = self._load_config()

    @classmethod
    def get_instance(cls) -> "NovaConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_config(self) -> HazardConfig:
        try:
            raw_text = self._config_path.read_text(encoding="utf-8")
            config_data = json.loads(raw_text)
            return HazardConfig(**config_data)
        except FileNotFoundError:
            logger.warning("Hazard config file not found at %s. Using defaults.", self._config_path)
            return HazardConfig()
        except json.JSONDecodeError as exc:
            logger.warning("Invalid hazard config JSON at %s: %s. Using defaults.", self._config_path, exc)
            return HazardConfig()
        except ValidationError as exc:
            logger.warning("Hazard config validation failed: %s. Using defaults.", exc)
            return HazardConfig()
        except Exception as exc:
            logger.warning("Unexpected error loading hazard config: %s. Using defaults.", exc)
            return HazardConfig()

    def get_config(self) -> HazardConfig:
        """Return the cached hazard configuration."""
        return self._config

    def reload(self) -> HazardConfig:
        """Reload configuration from disk and replace the cached values."""
        self._config = self._load_config()
        return self._config
