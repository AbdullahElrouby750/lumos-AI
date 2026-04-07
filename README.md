# Lumos: Social Vision & Face Recognition Module (Nova Team)

**Current Release: Version 2.0**
*(Note on Versioning: Major versions are tracked by official pushes to the GitHub remote repository. V2 represents the accumulation of recent local Git commits pushed to remote, featuring the major "Demo-Proof" architectural overhaul).*

## 👁️ Project Overview
This repository contains the **Social Vision Module** for **Lumos**, an assistive technology wearable designed for the visually impaired. 

Moving beyond standard obstacle detection, this module acts as a "Social Brain". It actively scans the environment, detects human faces, calculates their distance, identifies known individuals using Deep Learning, and provides real-time, context-aware audio feedback to the user.

### The Prime Directive: Zero Blocking
Because this is assistive technology for the visually impaired, **camera freezing or frame dropping is a critical safety hazard**. This entire architecture is built around a "Prime Directive": The main camera loop must *never* drop below 30 FPS. All heavy processing (DeepFace alignment, Text-to-Speech audio drivers, network listening, and state machines) are offloaded to asynchronous background threads and queues.

---

## ⚙️ How It Works (The V2 Architecture)
The system uses a highly optimized **Microservice-style CEO/Worker Architecture**:
1. **The CEO (`main.py`):** Captures frames at 30 FPS and delegates them.
2. **The Consensus Gate (Vision Pipeline):** Uses a lightweight `CentroidTracker` to follow moving bounding boxes. When a face is detected, it spawns a worker thread to align and identify the face. It requires a **3-frame Temporal Vote** before locking in a result to prevent hallucination from motion blur.
3. **The Jitter Buffer & Collision Zone:** Calculates a central 40% "Collision Zone," ignoring people passing safely on the periphery. It utilizes a **5-frame Moving Average Buffer** to calculate bounding box expansion, eliminating camera jitter and only warning the user if a recognized face is physically approaching them. Alerts are locked for 5 seconds to prevent spam.
4. **The Wake Word NLP Engine:** Uses a UDP socket to listen for voice commands. Commands are locked behind a fuzzy-matched **Wake Word Gate** ("Lumo", "Luma", etc.) so background conversations are ignored.
5. **The 7-Rank Audio Manager:** A thread-safe Priority Queue ensures the Text-to-Speech engine never crashes. It categorizes audio into 7 strict ranks (Critical down to Low). If a voice command is received, the system executes a "Barge-In," pausing lower-priority speech instantly so the user can be heard.

---

## 📁 Module Breakdown

### 1. Core Orchestration
#### `main.py` (The CEO)
The entry point of the application. It initializes the camera, spins up the background threads (Audio Queue, Command Listener), and runs the master `while` loop. It routes frames to the Vision Pipeline and the Enrollment Manager. 

#### `utils.py` (Math, Tracking & State Management)
Contains pure mathematical logic and command processing.
* `CentroidTracker`: Calculates Euclidean distance between bounding box centers across frames to maintain temporary IDs for faces.
* `is_bbox_expanding()`: Calculates a moving average of face widths over recent frames to definitively prove an object is approaching.
* `process_command()`: Handles intent routing. Includes a memory state machine that uses `difflib` for fuzzy matching names and requires explicit "Yes/No" verbal confirmation before executing destructive actions like forgetting a user.

### 2. Vision & Recognition
#### `nova_vision_pipeline.py` (The Security Guard)
Orchestrates the detection and recognition process. Manages the async `recognize_worker` threads, enforcing the 3-frame consensus vote. It also manages the 5-second `alerted_approaching_ids` lock and triggers `voice_queue.speak()`.

#### `nova_face_detector.py`
A lightweight wrapper around Google's `MediaPipe` Face Detector (`blaze_face_short_range.tflite`). Returns precise bounding boxes and landmarks.

#### `nove_face_rec.py`
The heavy recognition engine utilizing `DeepFace` and `Facenet512`.
* **How it works:** When called on a live crop, it forces `align=True` to mathematically straighten the face before extracting the embedding. It calculates Cosine Distance against the known database with a strict 0.4 threshold.

### 3. Identity Management (The Brain)
#### `nova_enrollment.py` (The Guided State Machine)
Handles the addition of new faces asynchronously.
* **How it works:** Acts as a state machine that verbally guides the user through a biometric scan. Features a **Visibility Validation Gate** that uses MediaPipe to ensure the camera can actually see a face before capturing a frame, resetting the timer if the face is lost. 

#### `nove_face_encoder.py`
Uses DeepFace (with forced alignment) to extract 512-dimensional embeddings for the newly captured images, appends them to `nova_brain.pkl`, and deletes the raw images to protect user privacy.

#### `nove_forget.py`
Safely loads the `.pkl` brain, deletes the target's embeddings, and overwrites the file.

### 4. Audio & Voice Input
#### `nova_audio.py` (The Voice Queue)
A bulletproof, thread-safe Producer-Consumer queue for `pyttsx3`.
* **Features:** 7-Rank hierarchy (1: Critical to 7: Low) with a 7-second Time-to-Live (TTL). Includes `pause_below_critical()` for instant voice barge-in and per-rank spam cooldowns (e.g., 10 seconds for Social greetings).

#### `nova_listener.py` (The Ear)
A UDP socket server running on a background daemon thread (Port `65432`). Listens for JSON payloads and triggers the voice queue pause mechanisms upon receiving a command.

#### `nova_commands.py` (The NLP Engine)
A custom Natural Language Processor. Features the `_contains_wake_word()` gate to filter out background noise. Isolates keywords, extracts target entities, and returns structured intents (e.g., `INTENT_CONFIRM`, `INTENT_CANCEL`).

---

## 🚀 How to Run

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install opencv-python mediapipe deepface pyttsx3 numpy SpeechRecognition tf-keras