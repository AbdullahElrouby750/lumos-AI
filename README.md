# Lumos: Assistive Wearable AI
**Current Release: Version 3.1 (The "Distributed Brain" Update)** **Status:** Transitioning to a Decoupled Edge-Server Architecture optimized for Raspberry Pi 4.

## 👁️ Project Overview
Lumos is a real-time assistive wearable AI designed to provide environmental and social autonomy for the visually impaired. It transforms raw video into social context, detecting faces, identifying individuals, reading text, and managing spatial proximity alerts without ever dropping below 30 FPS.

## ⚙️ V3.1 Architectural Breakthroughs

Version 3.1 solves the "Blocking Problem" of heavy AI models on edge hardware by splitting continuous reflexes and deep thinking into completely isolated, thread-safe workers.

### 1. The Decoupled Worker Architecture
* **The Continuous Reflex (YOLOv11):** Obstacle and vehicle detection runs in a dedicated daemon thread. It consumes frames via a bounded queue and uses a semaphore lock, ensuring it never throttles the main camera feed.
* **The On-Demand Brain (Gemini 1.5 & EasyOCR):** Heavy cloud and localized AI models sit entirely asleep until triggered by explicit user intents (e.g., `INTENT_SCENE`). If the mobile hotspot drops, strict timeouts catch the failure gracefully instead of freezing the system.
* **Zero-SD Card Wear (RAM Disk):** High-resolution image captures for Gemini and OCR are piped directly to `/dev/shm/latest_scene.jpg` (RAM), ensuring zero read/write degradation to the physical SD card.

### 2. The Unified Nervous System
* **FastAPI & WebSockets:** Replaced synchronous loops with an ASGI-native server handling high-speed JSON event multiplexing.
* **Priority-Ranked Signaling:** All system alerts are mapped to a unified 7-rank priority queue (from `PRIORITY_CRITICAL` for fast-moving vehicles to `PRIORITY_LOW` for background scene descriptions).
* **RAM-Cached Hot Reloads:** Configuration for danger distances and hazards is read once into a memory cache from `hazards_config.json`, allowing for instantaneous parameter updates without disk I/O during the 30 FPS loop.

---

## 📁 Production Folder Structure

```text
lumos-v3/
├── config/                     # System configuration
│   ├── hazards_config.json     # Danger thresholds and objects
│   └── nova_config_manager.py  # RAM Cache singleton
├── data/                       # Persistent localized data
│   └── face_db/                # DeepFace embedded identities
├── models/                     # Heavy AI weights
│   └── yolo11n.pt              # Ultralytics model
├── src/
│   ├── core/                   # The Pipeline & Base Logic
│   │   ├── nova_vision_pipeline.py # The CEO (Main Loop)
│   │   └── nova_commands.py        # Intents & NLP Logic
│   ├── network/                # FastAPI & Pydantic
│   │   ├── nova_server.py
│   │   └── nova_network_models.py
│   ├── workers/                # Isolated Thread Workers
│   │   ├── nova_yolo_worker.py
│   │   └── nova_ai_worker.py
│   └── modules/                # Specialized Task Logic
│       ├── brain_module.py
│       └── OCR.py
└── main.py                     # Entry Point

🚀 Deployment Instructions
1. Environment Setup
Ensure you are using Python 3.10+ and install the dependencies:

python -m venv .venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt

2. Download YOLO Weights
Fetch the required edge model:
python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"

3. Launch the System
python main.py

Nova & Lumos Team | 2026 Graduation Project