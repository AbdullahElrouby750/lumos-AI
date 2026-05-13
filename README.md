

# Lumos: Assistive Wearable AI

**Current Release: Version 3.1 (The "Nervous System" Update)** **Status:** Transitioning to a Distributed Edge-Server Architecture.

## 👁️ Project Overview

Lumos is a real-time assistive wearable AI designed to provide environmental and social autonomy for the visually impaired. Unlike traditional assistive tools, Lumos uses a **Distributed Hub Model**:

* **The Eyes (Raspberry Pi 4):** Handles high-frequency computer vision, face recognition (Nova), and obstacle detection.
* **The Voice (Mobile App):** Acts as the primary user interface, handling Text-to-Speech (TTS), GPS navigation, and high-level AI reasoning (Gemini).
* **The Bridge (WebSocket):** A high-speed asynchronous connection over a mobile hotspot that ensures instant haptic and audio feedback.

---

## 🚀 Key Features

### 1. Social Vision & Identity (Nova Core)

* **DeepFace Identification:** Recognizes known individuals using Facenet512 with anatomical pose validation.
* **Proximity Awareness:** Tracks multiple individuals simultaneously, providing distance-relative spatial alerts.
* **T-Zone Pose Math:** Mathematically verifies if a person is looking at the user before speaking, preventing unnecessary interruptions.

### 2. Safety & Scene Intelligence (Luma Integration)

* **Obstacle Detection:** Real-time YOLO-based detection of cars, trip hazards, and crowds.
* **Emergency Overrides:** Instant vocal and haptic alerts for fast-moving vehicles within a 4-meter radius.
* **Gemini Scene Description:** Multi-modal AI that describes complex surroundings, such as "A cozy cafe with three people sitting to your left."
* **OCR & Document Reading:** Extracting text from menus, signs, and labels via EasyOCR and Gemini fallbacks.

### 3. Smart Navigation

* **Turn-by-Turn Guidance:** Integration with OpenRouteService for pedestrian-optimized routing.
* **Phone-Linked GPS:** Utilizes the high-precision GPS sensors of the user's smartphone via the Lumos mobile app.

---

## 🛠️ V3.1 Architectural Breakthroughs (The "Nervous System")

This version marks the shift from a monolithic application to an **Edge-Server model**, optimized specifically for the **Raspberry Pi 4**.

### 1. Unified Asynchronous Backend

* **FastAPI & Uvicorn:** Replaced traditional threading loops with an ASGI-native server handling both WebSockets and REST API on Port 5000.
* **Zero-Config Discovery (mDNS):** Implemented Zeroconf/Bonjour discovery. The mobile app automatically finds `lumos.local` on the network without requiring a static IP.

### 2. Hardware Offloading & Optimization

* **Vocal Offloading:** Text-to-Speech generation has been moved to the mobile app, saving significant CPU cycles and RAM on the Pi.
* **RAM-Disk I/O (`/dev/shm`):** High-resolution captures for Gemini are stored in the Pi's RAM disk rather than the SD card, increasing speed and extending hardware lifespan.
* **Event Multiplexing:** A single WebSocket pipe handles all communication using a structured JSON/MessagePack event system.

### 3. The Sync-to-Async Bridge

* **Thread-Safe Signaling:** Implemented an `asyncio.Queue` bridge that allows the synchronous Vision Pipeline to push detection events to the asynchronous network server without blocking the 30 FPS camera feed.

---

## 📁 System Modules

* **`nova_server.py`**: The FastAPI core managing real-time signaling.
* **`nova_discovery.py`**: Handles mDNS service registration for "Zero-Touch" pairing.
* **`nova_vision_pipeline.py`**: The "CEO" script managing the camera and AI workers.
* **`brain_module.py`**: Interface for Gemini 1.5 Flash vision capabilities.
* **`ORS.py` / `OCR.py**`: Specialized modules for navigation and text recognition.

---

## 📥 Deployment

1. **Configure Hotspot:** Set your mobile phone to Hotspot mode.
2. **Start the Server:**
```bash
python nova_server.py

```


3. **Launch Vision:**
```bash
python main.py

```


4. **Connect:** Open the Lumos Mobile App; it will automatically discover and pair with the glasses.

**Nova & Luma Team | 2026 Graduation Project**
