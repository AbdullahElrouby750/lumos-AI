

# Lumos AI — Wearable Assistive Ecosystem

Lumos is an advanced, headless AI-driven edge wearable ecosystem designed to provide comprehensive situational awareness, spatial navigation, and social integration for visually impaired individuals. 

Built on a distributed asymmetric thread-safe architecture, Lumos transforms a wearable processor (such as a Raspberry Pi 4) into a silent local node that handles low-latency sensory processing (30 FPS) while isolating heavy intelligence operations into autonomous background worker threads, communicating seamlessly with a mobile client via structured network protocol streams.


## 🏗️ Architectural Core: The Asymmetric Pipeline

Lumos rejects the traditional monolithic loop architecture. It utilizes a central orchestrator (`main.py`) acting as the "CEO," managing an interconnected system of non-blocking workers and networks. 



```
                   ┌─────────────────────────┐
                   │  Mobile Client (Phone)  │
                   └────────────┬────────────┘
                                │  ▲
                     WebSockets │  │ REST API
                                ▼  │

┌────────────────────────────────────┴─────────────────────────────────────┐
│ Lumos Headless Server (FastAPI Core)                                     │
└─────────────────┬──────────────────────────────────────▲─────────────────┘
│                                      │
┌─────────────────▼──────────────────────────────────────┴─────────────────┐
│ VisionPipeline (The CEO Core Orchestrator)                               │
└───────┬──────────────────────┬──────────────────────┬─────────────▲──────┘
│                      │                      │             │
▼ Enqueue Frame        ▼ Enqueue Command      ▼ Enqueue     │ Callback
┌───────────────┐      ┌───────────────┐      ┌───────────────┐     │ (Hot-Reload)
│  YOLO Worker  │      │   AI Worker   │      │  Enrollment   │─────┘
│   (Hazards)   │      │  (Scene/OCR)  │      │    Worker     │
└───────────────┘      └───────────────┘      └───────────────┘

```

### 1. The Nervous System (`src/network/`)
* **`LumosServer`**: An asynchronous FastAPI server core managed on Uvicorn. It multiplexes real-time bidirectional WebSocket event channels (`/ws`) and exposes high-speed RESTful API streams (`/api/v1/scene`, `/api/v1/brain`) utilizing a local memory RAM disk (`/dev/shm`) to prevent flash storage degradation.
* **`LumosDiscovery`**: Zero-touch networking powered by mDNS/Zeroconf. It allows the wearable node to automatically broadcast its coordinates on the local network, enabling instant pairing with the companion mobile application without user configuration.

### 2. The Reflex Loop (`src/core/`)
* **`VisionPipeline`**: The primary camera driver and feature dispatcher running synchronously at 30 FPS. It combines high-speed localized facial landmark tracking with tactical tracking constraints to compute depth and relative velocities.
* **`VoiceQueue`**: A thread-safe, 7-rank priority output queue implementing proactive barge-in overrides, Time-To-Live (TTL) packet pruning, and per-rank cooling windows to prevent alert fatigue. It instantly serializes text arrays into networking data payloads, deferring the physical Text-to-Speech execution to the client hardware.

### 3. Asynchronous Worker Daemons (`src/workers/`)
* **`NovaYoloWorker`**: An isolated, continuous object and hazard detection thread. It evaluates spatial threat matrices (e.g., expanding collision vectors, drop-offs) without penalizing the target camera frame rate.
* **`NovaAIWorker`**: A state-machine tracking high-level intent. It handles complex, high-latency multimodal vision tasks like broad scene descriptions and optical character recognition (OCR) by executing remote API calls completely off the main thread.
* **`NovaEnrollmentWorker`**: A dedicated face enrollment state machine. It handles multi-axis head pose tracking and validation, running heavy DeepFace database mathematical representations asynchronously, then hot-reloads the master brain via a thread-safe feedback callback.

---

## 🧠 Key Capability Subsystems (`src/modules/`)

* **Facial Analysis & Biometrics (`nove_face_rec`, `nova_face_detector`)**: Leverages real-time MediaPipe face landmark configurations coupled with an asynchronous DeepFace `Facenet512` model processing pipeline. Tracks persistent identities via custom Centroid Tracking algorithms to isolate target features.
* **Pose & Orientation Gating (`nova_pose_validator`)**: Utilizes moving-average smoothing matrices and consensus frame counters to validate head trajectories (Pitch, Yaw, Roll) during interactive setup sequences.
* **Brain Topography (`brain_module`, `nove_face_encoder`)**: Compiles individual biometric matrices into a local serialized storage cluster (`nova_brain.pkl`) while keeping standard image sets ephemeral.
* **Tactical Navigation (`hazards`)**: Processes configuration maps (`hazards_config.json`) specifying horizontal collision ranges to identify nearby obstacles, vehicles, or topological drops.

---

## 🛠️ Thread Safety & Memory Design

Lumos enforces strict architectural rules to maximize battery and thermal efficiency on low-power edge units like the Raspberry Pi 4:

1.  **Memory Decoupling**: Frames sent to background worker threads are explicitly deep-copied (`frame.copy()`). This prevents memory collisions where the main thread modifies pixel arrays while background workers execute matrix multiplications.
2.  **Serialized Mutation**: Core pipeline dictionaries (`recognition_results`, `temporal_votes`, `unknown_timers`) are owned solely by the main execution stream. Any auxiliary thread modifications must serialize through a state synchronization lock (`self._state_lock`).
3.  **Zero Thread Leaks**: All daemon engines implement standard threading events (`_stop_event`). When a termination vector (`INTENT_QUIT` or manual kill switch) is received, the system enforces sequential teardown closures across all network ports and C++ memory allocations.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* OpenCV dependencies (`libgl1-mesa-glx`)
* Hardware video device mapped to `/dev/video0`

### Installation
1.  Clone the repository down to your local directory:
    ```bash
    git clone [https://github.com/abdullahelrouby750/lumos-ai.git](https://github.com/abdullahelrouby750/lumos-ai.git)
    cd lumos-ai
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Execution
Boot the headless core processor:
```bash
python main.py

```

Upon execution, the server will open port `5000`, deploy mDNS broadcast markers, initialize local machine learning networks, and await structured incoming JSON payloads from the accompanying client interface.

```